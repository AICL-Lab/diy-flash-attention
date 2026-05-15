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


def test_learning_path_pages_exist_and_reference_core_docs() -> None:
    en = (ROOT / "docs/learning-path.md").read_text()
    zh = (ROOT / "docs/zh/learning-path.md").read_text()

    for text in ("tutorial", "architecture", "algorithm", "performance", "api"):
        assert text in en.lower()

    for text in ("教程", "架构", "算法", "性能", "api"):
        assert text in zh.lower()

    for stage in (
        "## Stage 1",
        "## Stage 2",
        "## Stage 3",
        "## Stage 4",
        "## Stage 5",
    ):
        assert stage in en

    assert "# 学习路径" not in en
    assert "## 阶段 1" not in en
    assert en != zh


def test_paper_guide_pages_cover_core_papers() -> None:
    en = (ROOT / "docs/paper-guide.md").read_text()
    zh = (ROOT / "docs/zh/paper-guide.md").read_text()

    assert "FlashAttention" in en
    assert "FlashAttention-2" in en
    assert "what to read first" in en.lower()
    assert "](/algorithm)" in en

    assert "FlashAttention" in zh
    assert "FlashAttention-2" in zh
    assert "先读" in zh
    assert "](/zh/algorithm)" in zh


def test_homepages_prioritize_academy_portal_sections() -> None:
    en = (ROOT / "docs/index.md").read_text()
    zh = (ROOT / "docs/zh/index.md").read_text()

    assert "## Start Here" in en
    assert "Learning Path" in en
    assert "Paper Guide" in en
    assert "Knowledge Map" in en
    assert "Reference Library" in en
    assert "## Why FlashAttention?" not in en
    assert "## Key Features" not in en
    assert "## Quick Start" not in en
    assert "## GPU Support Matrix" not in en

    assert "## 从这里开始" in zh
    assert "学习路径" in zh
    assert "论文导读" in zh
    assert "知识图谱" in zh
    assert "参考资料库" in zh
    assert "知识地图" not in zh
    assert "## FlashAttention 架构" not in zh
    assert "## 核心特性" not in zh
    assert "## 快速开始" not in zh
    assert "## GPU 支持矩阵" not in zh


def test_homepage_portal_links_do_not_hardcode_repo_base() -> None:
    en = (ROOT / "docs/index.md").read_text()
    zh = (ROOT / "docs/zh/index.md").read_text()

    assert "/diy-flash-attention/" not in en
    assert "/diy-flash-attention/" not in zh


def test_static_pwa_assets_do_not_hardcode_repo_base() -> None:
    for relative_path in (
        "docs/public/manifest.json",
        "docs/public/sw.js",
        "docs/public/offline.html",
    ):
        text = (ROOT / relative_path).read_text()
        assert "/diy-flash-attention/" not in text, relative_path


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
    assert "import.meta.env.BASE_URL" in theme_index
    assert "'/diy-flash-attention'" not in theme_index
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
