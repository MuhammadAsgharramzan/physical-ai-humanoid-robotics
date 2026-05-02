# Physical AI E-Book EPUB Generator

This tool crawls the [Physical AI & Humanoid Robotics](https://physical-ai-humanoid-robotics-tan.vercel.app) website and generates a professionally formatted EPUB file for offline reading on Kindle, Apple Books, or other e-readers.

## Prerequisites

- **Node.js** (v16 or higher)
- **npm** (comes with Node.js)

## Installation

1. Navigate to the `epub-generator` directory:
   ```bash
   cd epub-generator
   ```

2. Install the required dependencies:
   ```bash
   npm install
   ```

## Usage

Run the generator script:
```bash
node generate-epub.js
```

The script will:
1. Launch a headless browser using **Puppeteer**.
2. Visit the introductory page and extract all lesson links from the sidebar.
3. Visit each page, extract the main content, and clean it (removing navigation, footers, and scripts).
4. Bundle everything into a single EPUB file named `physical-ai-humanoid-robotics.epub`.

## Customization

You can modify the `css` property in `generate-epub.js` to change the appearance of the generated e-book (e.g., fonts, colors, or code block styling).
