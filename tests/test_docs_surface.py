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


def test_logo_svg_declares_theme_adaptive_rules() -> None:
    logo = (ROOT / "docs/public/logo.svg").read_text()
    assert "prefers-color-scheme" in logo
    assert "--fa-logo-bg" in logo
    assert "--fa-logo-fg" in logo


def test_theme_aware_figure_component_is_registered() -> None:
    theme_index = (ROOT / "docs/.vitepress/theme/index.ts").read_text()
    component = (ROOT / "docs/.vitepress/theme/components/ThemeAwareFigure.vue").read_text()
    assert "ThemeAwareFigure" in theme_index
    assert "defineProps" in component
    assert "isDark" in component
    assert "withBase" in component
    assert "normalizeSrc" in component
    assert "data:" in component
    assert "startsWith('/')" in component or "startsWith(\"/\")" in component
    assert "replace(/^(?:\\.\\.\\/|\\.\\/)+/, '')" in component or "replace(/^(?:\\.\\.\\/|\\.\\/)+/, \"\")" in component


def test_docs_config_uses_theme_aware_logo_assets() -> None:
    config = CONFIG.read_text()
    assert "logo: {" in config
    assert "light: withBasePath('/logo-light.svg')" in config
    assert "dark: withBasePath('/logo-dark.svg')" in config
    assert "src: withBasePath('/logo.svg')" not in config
