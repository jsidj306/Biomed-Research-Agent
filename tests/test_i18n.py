"""
UI 国际化（i18n）— 单元测试。

覆盖 src/ui/i18n.py 的 t() 翻译函数与文案字典完整性。
纯函数逻辑，无外部依赖。
"""

from src.ui.i18n import SUPPORTED_LANGUAGES, TEXT, t

# ── 基本翻译 ─────────────────────────────────────────────────────────────


def test_t_returns_chinese():
    assert t("zh", "title") == "🧬 生物医学研究助手"


def test_t_returns_english():
    assert t("en", "title") == "🧬 Biomedical Research Agent"


def test_t_zh_and_en_differ():
    assert t("zh", "config") != t("en", "config")


# ── 占位符替换 ────────────────────────────────────────────────────────────


def test_t_placeholder_substitution():
    assert t("zh", "found_n_papers", count=5, shown=3) == "**找到 5 篇论文** — 显示前 3 篇"
    assert t("en", "searching", sources="PubMed", query="cancer").startswith("🔍 Searching")


# ── 回退行为 ──────────────────────────────────────────────────────────────


def test_t_unknown_lang_falls_back_to_english():
    assert t("fr", "title") == t("en", "title")


def test_t_unknown_key_returns_key_itself():
    assert t("zh", "this_key_does_not_exist") == "this_key_does_not_exist"


# ── 字典完整性 ────────────────────────────────────────────────────────────


def test_supported_languages():
    assert set(SUPPORTED_LANGUAGES) == {"zh", "en"}


def test_zh_en_have_same_keys():
    """中英文文案键必须一一对应，避免某语言漏译。"""
    assert set(TEXT["zh"]) == set(TEXT["en"])


def test_all_templates_resolvable_without_args():
    """无占位符的文案应能直接取值；含占位符的由测试单独覆盖。"""
    for lang in ("zh", "en"):
        for template in TEXT[lang].values():
            assert isinstance(template, str)
            assert template  # 非空
