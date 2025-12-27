import os
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import markdown
from bs4 import BeautifulSoup
import frontmatter

from .content_indexer import content_indexer

logger = logging.getLogger(__name__)

class DocusaurusContentPipeline:
    def __init__(self, docs_path: str = "../../../docs"):
        self.docs_path = Path(docs_path)
        self.supported_extensions = ['.md', '.mdx']

    def _extract_content_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content and metadata from a markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse frontmatter and content
            post = frontmatter.loads(content)
            metadata = post.metadata
            markdown_content = post.content

            # Convert markdown to plain text for indexing
            html = markdown.markdown(markdown_content)
            soup = BeautifulSoup(html, 'html.parser')
            plain_text = soup.get_text()

            # Extract module and lesson info from path
            path_parts = file_path.relative_to(self.docs_path).parts
            module_id = path_parts[0] if len(path_parts) > 1 else None
            lesson_id = file_path.stem if path_parts else None

            return {
                "title": metadata.get('title', file_path.stem),
                "content": plain_text,
                "module_id": module_id,
                "lesson_id": lesson_id,
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return None

    async def index_all_content(self) -> bool:
        """Index all markdown content in the docs directory"""
        try:
            indexed_count = 0
            failed_count = 0

            for ext in self.supported_extensions:
                for file_path in self.docs_path.rglob(f"*{ext}"):
                    content_data = self._extract_content_from_file(file_path)

                    if content_data:
                        content_id = f"{content_data['module_id']}_{content_data['lesson_id']}"
                        success = await content_indexer.index_content(
                            content_id=content_id,
                            title=content_data["title"],
                            content=content_data["content"],
                            module_id=content_data["module_id"],
                            lesson_id=content_data["lesson_id"]
                        )

                        if success:
                            indexed_count += 1
                            logger.info(f"Successfully indexed: {content_id}")
                        else:
                            failed_count += 1
                            logger.error(f"Failed to index: {content_id}")
                    else:
                        failed_count += 1
                        logger.error(f"Failed to extract content from: {file_path}")

            logger.info(f"Indexing complete. Indexed: {indexed_count}, Failed: {failed_count}")
            return failed_count == 0

        except Exception as e:
            logger.error(f"Error in indexing pipeline: {e}")
            return False

    async def index_single_file(self, file_path: str) -> bool:
        """Index a single markdown file"""
        try:
            path = Path(file_path)
            content_data = self._extract_content_from_file(path)

            if content_data:
                content_id = f"{content_data['module_id']}_{content_data['lesson_id']}"
                success = await content_indexer.index_content(
                    content_id=content_id,
                    title=content_data["title"],
                    content=content_data["content"],
                    module_id=content_data["module_id"],
                    lesson_id=content_data["lesson_id"]
                )
                return success
            return False
        except Exception as e:
            logger.error(f"Error indexing single file {file_path}: {e}")
            return False

# Create pipeline instance
content_pipeline = DocusaurusContentPipeline()