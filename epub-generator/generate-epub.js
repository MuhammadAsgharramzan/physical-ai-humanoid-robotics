const puppeteer = require('puppeteer');
const epubGen = require('epub-gen-memory').default;
const path = require('path');
const fs = require('fs');

const BASE_URL = 'https://physical-ai-humanoid-robotics-tan.vercel.app';
const START_URL = `${BASE_URL}/docs/intro`; 
const OUTPUT_FILE = path.join(__dirname, 'physical-ai-humanoid-robotics.epub');

async function generateEpub() {
    console.log('Starting EPUB generation process...');
    
    const browser = await puppeteer.launch({
        headless: "new"
    });
    const page = await browser.newPage();

    console.log(`Fetching navigation from: ${START_URL}`);
    await page.goto(START_URL, { waitUntil: 'networkidle2' });
    
    const links = await page.evaluate((baseUrl) => {
        const menuLinks = Array.from(document.querySelectorAll('.menu__link'));
        return menuLinks
            .map(link => link.href)
            .filter(href => href.startsWith(baseUrl) && !href.includes('#'))
            .filter((value, index, self) => self.indexOf(value) === index);
    }, BASE_URL);

    console.log(`Found ${links.length} unique chapters to include.`);

    const content = [];

    for (const url of links) {
        console.log(`Processing: ${url}`);
        try {
            await page.goto(url, { waitUntil: 'networkidle2' });
            
            const pageData = await page.evaluate(() => {
                const article = document.querySelector('article');
                if (!article) return null;

                const selectorsToRemove = [
                    'nav', 'footer', '.theme-doc-footer', 
                    '.theme-doc-breadcrumb', '.table-of-contents',
                    'button', 'script', 'style', 'iframe', 'header'
                ];
                selectorsToRemove.forEach(s => {
                    article.querySelectorAll(s).forEach(el => el.remove());
                });

                const title = article.querySelector('h1')?.innerText || 'Untitled Section';
                
                article.querySelectorAll('img').forEach(img => {
                    if (img.getAttribute('src').startsWith('/')) {
                        img.src = window.location.origin + img.getAttribute('src');
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
                    content: pageData.data
                });
            }
        } catch (err) {
            console.error(`Failed to process ${url}:`, err.message);
        }
    }

    await browser.close();

    const options = {
        title: "Physical AI & Humanoid Robotics",
        author: "Tan",
        publisher: "Physical AI Textbook",
        css: `
            body { font-family: sans-serif; line-height: 1.6; padding: 20px; }
            h1 { color: #2e8555; text-align: center; margin-top: 40px; }
            h2 { color: #3578e5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }
            pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; font-family: monospace; font-size: 0.85em; border: 1px solid #dfe1e4; overflow-x: auto; }
            code { background-color: rgba(175, 184, 193, 0.2); padding: 0.2em 0.4em; border-radius: 3px; }
            img { max-width: 100%; height: auto; display: block; margin: 20px auto; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { border: 1px solid #dfe1e4; padding: 10px; text-align: left; }
            blockquote { border-left: 4px solid #dfe1e4; padding: 10px 20px; color: #666; font-style: italic; background: #fafafa; }
        `
    };

    console.log('Building EPUB file...');
    try {
        const buffer = await epubGen(options, content);
        fs.writeFileSync(OUTPUT_FILE, buffer);
        console.log(`Successfully generated EPUB: ${OUTPUT_FILE}`);
    } catch (err) {
        console.error("Failed to generate EPUB:", err);
    }
}

generateEpub();
