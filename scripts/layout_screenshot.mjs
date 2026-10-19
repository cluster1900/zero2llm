// Return whether a screenshot was captured; absence of figures is not a layout error.
export async function capturePhoneFigure(page, outputPath) {
  const figure = await page.$('.diagram-figure, figure');
  if (!figure) return false;
  try {
    await figure.evaluate(el => el.scrollIntoView({block: 'start'}));
    await page.screenshot({path: outputPath});
    return true;
  } finally {
    await figure.dispose();
  }
}
