// Explicit directory entry for `node --test tests/` on Node versions that do not
// expand a directory argument. This imports the real tests without changing them.
import('./layout_screenshot.test.mjs');
