from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/.vitepress/config.mts"


def test_docs_nav_exposes_academy_routes() -> None:
    text = CONFIG.read_text()
    assert "/learning-path" in text
    assert "/paper-guide" in text
    assert "/knowledge-map" in text
    assert "/zh/learning-path" in text
    assert "/zh/paper-guide" in text
    assert "/zh/knowledge-map" in text


def test_docs_academy_route_files_exist_for_english() -> None:
    assert (ROOT / "docs/learning-path.md").exists()
    assert (ROOT / "docs/paper-guide.md").exists()
    assert (ROOT / "docs/knowledge-map.md").exists()


def test_docs_academy_route_files_exist_for_chinese() -> None:
    assert (ROOT / "docs/zh/learning-path.md").exists()
    assert (ROOT / "docs/zh/paper-guide.md").exists()
    assert (ROOT / "docs/zh/knowledge-map.md").exists()


def test_docs_config_uses_computed_base_variable() -> None:
    text = CONFIG.read_text()
    assert "const base =" in text
    assert "base: base" in text
