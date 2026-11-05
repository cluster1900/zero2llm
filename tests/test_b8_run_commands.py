"""B8: a chapter's run block runs that chapter's test file, not the whole suite."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "chapters"
TESTS = ROOT / "tests"

OLD_CMD = "python3 -m unittest discover -s tests -v"
OLD_SENTENCE = "运行上面的测试命令，末尾应显示 `OK`；本章相关的测试分类见 `docs/testing.md`。"
NEW_SENTENCE = "运行本章测试文件，末尾应显示 `OK`；测试分类、计数与错误版本清单见 `docs/testing.md`。"
PATTERN = re.compile(r"-p\s+'?(test_\w+\.py)'?")
SKIP = {"00_preface.md", "10_appendix.md"}

CHANGED = {
    "01_word_embedding.md": [
        (OLD_CMD, "python3 -m unittest discover -s tests -p 'test_foundations.py' -v"),
        (OLD_SENTENCE, NEW_SENTENCE),
    ],
    "01a_tokenizer.md": [
        (OLD_CMD, "python3 -m unittest discover -s tests -p 'test_foundations.py' -v"),
        (OLD_SENTENCE, NEW_SENTENCE),
    ],
    "03_qkv_self_attention.md": [
        (OLD_CMD, "python3 -m unittest discover -s tests -p 'test_attention.py' -v"),
        (OLD_SENTENCE, NEW_SENTENCE),
    ],
    "04_multi_head_attention.md": [
        (OLD_CMD, "python3 -m unittest discover -s tests -p 'test_attention.py' -v"),
    ],
    "07_encoder_decoder.md": [
        (OLD_CMD, "python3 -m unittest discover -s tests -p 'test_generation.py' -v"),
        (OLD_SENTENCE, NEW_SENTENCE),
    ],
}

KEPT = {
    "00_preface.md": [OLD_CMD],
    "10_appendix.md": [
        "仓库根目录的 `python -m unittest discover -s tests -v` 检查示例与契约，`python scripts/check_book.py` 检查已生成制品的资源/链接，`npm run check:layout` 检查正常浏览器视口布局。公式正确、链接存在、图片可读与代码通过互不替代。具体通过与跳过数以当批验证记录为准，不在这里保留会迅速过时的全书总数。",
    ],
}


def bash_command_lines(text):
    """Logical commands inside ```bash fences.

    A trailing backslash continues the shell command, so a wrapped
    `unittest discover` is still one command line and must carry `-p`.
    """
    in_fence = False
    pending = ""
    for line in text.splitlines():
        if not in_fence:
            if line.strip() == "```bash":
                in_fence = True
            continue
        if line.strip().startswith("```"):
            in_fence = False
            if pending:
                yield pending.strip()
                pending = ""
            continue
        piece = line.strip()
        if piece.endswith("\\"):
            pending += piece[:-1].rstrip() + " "
            continue
        yield (pending + piece).strip()
        pending = ""
    if pending:
        yield pending.strip()


class B8RunCommandsTest(unittest.TestCase):
    def test_changed_lines(self):
        for name, pairs in CHANGED.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            for old, new in pairs:
                with self.subTest(file=name, new=new[:40]):
                    self.assertEqual(body.count(old), 0)
                    self.assertEqual(body.count(new), 1)

    def test_kept_lines(self):
        for name, lines in KEPT.items():
            body = (CHAPTERS / name).read_text(encoding="utf-8")
            for line in lines:
                with self.subTest(file=name, line=line[:40]):
                    self.assertEqual(body.count(line), 1)

    def test_chapter_bash_discover_is_one_existing_file(self):
        commands = []
        for path in sorted(CHAPTERS.glob("*.md")):
            if path.name in SKIP:
                continue
            text = path.read_text(encoding="utf-8")
            for cmd in bash_command_lines(text):
                if "unittest discover" not in cmd:
                    continue
                commands.append((path.name, cmd))
                match = PATTERN.search(cmd)
                with self.subTest(file=path.name, cmd=cmd):
                    self.assertIsNotNone(match)
                    self.assertIn("-p", cmd)
                    self.assertTrue((TESTS / match.group(1)).is_file())
        self.assertGreaterEqual(len(commands), 1)


if __name__ == "__main__":
    unittest.main()
