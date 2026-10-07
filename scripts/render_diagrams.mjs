// Build-time rendering: reading the book never requires Mermaid or JavaScript.
import fs from 'node:fs/promises';
import path from 'node:path';
import puppeteer from 'puppeteer';
import { renderMermaid } from '@mermaid-js/mermaid-cli';

const [jobsFile, outputDir] = process.argv.slice(2);
if (!jobsFile || !outputDir) throw new Error('Usage: node scripts/render_diagrams.mjs jobs.json output-dir');
const jobs = JSON.parse(await fs.readFile(jobsFile, 'utf8'));
await fs.mkdir(outputDir, { recursive: true });
const browser = await puppeteer.launch({ headless: true });
const results = {};
try {
  for (const job of jobs) {
    let svg;
    if (job.definition) {
      const rendered = await renderMermaid(browser, job.definition, 'svg', {
        viewport: { width: 1200, height: 900, deviceScaleFactor: 2 },
        backgroundColor: '#ffffff',
        mermaidConfig: {
          theme: 'base',
          fontFamily: '"PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif',
          themeVariables: {
            fontSize: '18px', primaryColor: '#eff6ff', primaryTextColor: '#172554',
            primaryBorderColor: '#3b82f6', lineColor: '#475569',
            secondaryColor: '#ecfdf5', tertiaryColor: '#fff7ed',
          },
          flowchart: { htmlLabels: false, useMaxWidth: false, nodeSpacing: 22, rankSpacing: 28, padding: 12, wrappingWidth: 260 },
          sequence: { useMaxWidth: false, width: 125, actorMargin: 35, messageMargin: 28, wrap: true },
        },
      });
      svg = Buffer.from(rendered.data).toString('utf8');
    } else {
      svg = await fs.readFile(job.source, 'utf8');
    }
    const viewBox = svg.match(/viewBox="([^"]+)"/);
    if (!viewBox) throw new Error(`${job.id}: missing SVG viewBox`);
    const [, , width, height] = viewBox[1].split(/[\s,]+/).map(Number);
    if (!(width > 0 && height > 0)) throw new Error(`${job.id}: invalid dimensions`);
    // Explicit intrinsic dimensions prevent 100% x 100% SVGs from filling a reader page.
    svg = svg.replace(/<svg\b([^>]*)>/, (_, attrs) => {
      attrs = attrs.replace(/\s(?:width|height|style)="[^"]*"/g, '');
      return `<svg${attrs} width="${width}" height="${height}">`;
    });
    svg = svg.replace(/[\t ]+$/gm, '');
    await fs.writeFile(path.join(outputDir, `${job.id}.svg`), svg);
    const page = await browser.newPage();
    try {
      await page.setViewport({ width: Math.ceil(width), height: Math.ceil(height), deviceScaleFactor: 2 });
      await page.setContent(`<html><head><meta charset="utf-8"><style>html,body{margin:0;padding:0;background:white}svg{display:block}</style></head><body>${svg}</body></html>`);
      await page.evaluate(() => document.fonts.ready);
      await page.screenshot({ path: path.join(outputDir, `${job.id}.png`), type: 'png', fullPage: true });
    } finally {
      await page.close();
    }
    results[job.id] = { width, height };
    process.stdout.write(`  ✓ ${job.id}: ${Math.ceil(width)} × ${Math.ceil(height)}\n`);
  }
  await fs.writeFile(path.join(outputDir, 'dimensions.json'), JSON.stringify(results, null, 2) + '\n');
} finally {
  await browser.close();
}
