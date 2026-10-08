// Integration check against the built XHTML, with remote requests blocked.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import puppeteer from 'puppeteer';

const root = path.resolve(import.meta.dirname, '..');
const reportDir = process.argv[2] || await fs.mkdtemp(path.join(os.tmpdir(), 'zero2llm-layout-'));
await fs.mkdir(reportDir, { recursive: true });
const unpacked = path.join(reportDir, 'epub');
execFileSync('python3', ['-m', 'zipfile', '-e', path.join(root, 'dist/transformer_from_scratch.epub'), unpacked]);
const browser = await puppeteer.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.setRequestInterception(true);
  page.on('request', req => req.url().startsWith('file:') || req.url().startsWith('data:') ? req.continue() : req.abort());
  const chapters = (await fs.readdir(path.join(unpacked, 'EPUB/text'))).filter(n => /^ch\d+\.xhtml$/.test(n));
  let count = 0;
  for (const viewport of [{width: 390, height: 844}, {width: 768, height: 1024}, {width: 844, height: 390}]) {
    await page.setViewport(viewport);
    for (const chapter of chapters) {
      await page.goto(pathToFileURL(path.join(unpacked, 'EPUB/text', chapter)).href);
      const errors = await page.evaluate(() => Array.from(document.querySelectorAll('img')).flatMap(img => {
        const box = img.getBoundingClientRect();
        const ratio = img.naturalWidth / img.naturalHeight;
        // Border radii are cosmetic; the content dimensions must retain the original ratio.
        const fit = getComputedStyle(img).objectFit;
        const issues = [];
        if (!img.complete || !img.naturalWidth) issues.push('missing image');
        if (box.left < -1 || box.right > innerWidth + 1 || box.height > innerHeight * .71) issues.push('image overflow');
        if (fit !== 'contain' && Math.abs(box.width / box.height - ratio) > .03) issues.push('distorted image');
        return issues.map(issue => `${img.alt}: ${issue}`);
      }));
      if (errors.length) throw new Error(`${chapter} ${JSON.stringify(viewport)}: ${errors.join('; ')}`);
      count++;
      if (chapter === 'ch001.xhtml' && viewport.width === 390) {
        await page.$eval('.diagram-figure', el => el.scrollIntoView({block:'start'}));
        await page.screenshot({ path: path.join(reportDir, 'epub-phone.png') });
      }
    }
    await page.goto(pathToFileURL(path.join(root, 'dist/index.html')).href);
    const missing = await page.evaluate(() => [...document.images].filter(i => !i.complete || !i.naturalWidth).length);
    if (missing) throw new Error(`Web has ${missing} missing images`);
  }
  await page.setViewport({width: 1280, height: 960});
  const sizes = JSON.parse(await fs.readFile(path.join(root, 'dist/assets/rendered/dimensions.json'), 'utf8'));
  const cards = Object.keys(sizes).map(id => `<figure><img src="${pathToFileURL(path.join(root, `dist/assets/rendered/${id}.png`)).href}"><figcaption>${id}</figcaption></figure>`).join('');
  await page.setContent(`<html><head><meta charset="utf-8"><style>body{margin:16px;background:#e2e8f0;display:grid;grid-template-columns:repeat(4,1fr);gap:14px;font:16px sans-serif}figure{margin:0;background:white;padding:12px;text-align:center}img{width:100%;height:330px;object-fit:contain}figcaption{padding:8px}</style></head><body>${cards}</body></html>`, {waitUntil:'load'});
  await page.screenshot({ path: path.join(reportDir, 'all-diagrams.png'), fullPage:true });
  console.log(`PASS: ${count} chapter/viewport combinations; all Web images available offline. Screenshots: ${reportDir}`);
} finally {
  await browser.close();
}
