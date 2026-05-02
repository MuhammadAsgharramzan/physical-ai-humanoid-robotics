const puppeteer = require('puppeteer');
const Epub = require('epub-gen');
const cheerio = require('cheerio');
const path = require('path');

const BASE_URL = 'https://physical-ai-humanoid-robotics-tan.vercel.app';
const START_URL = `${BASE_URL}/docs/intro`; // Starting point for the textbook content
const OUTPUT_FILE = path.join(__dirname, 'physical-ai-humanoid-robotics.epub');

async function generateEpub() {
    console.log('Starting EPUB generation process...');
    
    const browser = await puppeteer.launch({
        headless: "new"
    });
    const page = await browser.newPage();

    // 1. Crawl the site to get all lesson links
    console.log(`Fetching navigation from: ${START_URL}`);
    await page.goto(START_URL, { waitUntil: 'networkidle2' });
    
    const links = await page.evaluate((baseUrl) => {
        const menuLinks = Array.from(document.querySelectorAll('.menu__link'));
        return menuLinks
            .map(link => link.href)
            .filter(href => href.startsWith(baseUrl) && !href.includes('#')) // Same domain, no fragments
            .filter((value, index, self) => self.indexOf(value) === index); // Unique links
    }, BASE_URL);

    console.log(`Found ${links.length} unique pages to include.`);

    const content = [];

    // 2. Fetch and clean content from each page
    for (const url of links) {
        console.log(`Processing: ${url}`);
        try {
            await page.goto(url, { waitUntil: 'networkidle2' });
            
            // Get the main content HTML
            const pageData = await page.evaluate(() => {
                const article = document.querySelector('article');
                if (!article) return null;

                // Remove unwanted elements
                const selectorsToRemove = [
                    'nav', 'footer', '.theme-doc-footer', 
                    '.theme-doc-breadcrumb', '.table-of-contents',
                    'button', 'script', 'style', 'iframe'
                ];
                selectorsToRemove.forEach(s => {
                    article.querySelectorAll(s).forEach(el => el.remove());
                });

                const title = article.querySelector('h1')?.innerText || 'Untitled Section';
                
                // Fix image paths to be absolute if they are relative
                article.querySelectorAll('img').forEach(img => {
                    if (img.src.startsWith('/')) {
                        img.src = window.location.origin + img.src;
                    }
                });

                return {
                    title: title,
                    data: article.innerHTML
                };
            });

            if (pageData) {
                content.push({
                    title: pageData.title,
                    data: pageData.data
                });
            }
        } catch (err) {
            console.error(`Failed to process ${url}:`, err.message);
        }
    }

    await browser.close();

    // 3. Generate the EPUB
    const option = {
        title: "Physical AI & Humanoid Robotics",
        author: "Tan",
        publisher: "Physical AI Textbook",
        output: OUTPUT_FILE,
        content: content,
        appendChapterTitles: false,
        verbose: true,
        css: `
            body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; }
            h1 { color: #2e8555; text-align: center; border-bottom: 2px solid #2e8555; padding-bottom: 10px; }
            h2 { color: #3578e5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }
            pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; font-family: 'Courier New', Courier, monospace; font-size: 0.9em; border: 1px solid #dfe1e4; }
            code { background-color: rgba(175, 184, 193, 0.2); padding: 0.2em 0.4em; border-radius: 6px; font-family: monospace; }
            img { max-width: 100%; height: auto; display: block; margin: 20px auto; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #dfe1e4; padding: 8px 12px; text-align: left; }
            th { background-color: #f6f8fa; }
            blockquote { border-left: 4px solid #dfe1e4; padding: 0 15px; color: #636c76; margin: 0; }
        `
    };

    console.log('Building EPUB file...');
    new Epub(option).promise.then(
        () => console.log(`Successfully generated EPUB: ${OUTPUT_FILE}`),
        err => console.error("Failed to generate EPUB:", err)
    );
}

generateEpub();
