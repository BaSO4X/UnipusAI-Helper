# -*- coding: utf-8 -*-
"""针对 v3.9.18 多小题简答题失败问题的回归测试。

覆盖三个修复点：
1. GenericReplyStrategy —— 多 textarea 页面枚举每个小题题干进 Question.text，
   让 AI 一开始就知道每个小题的存在与题干。
2. PromptBuilder._build_text —— 多空简答题在 prompt 中列出小题列表，
   并要求按题号逐题输出。
3. PromptBuilder._format_instructions —— 简答题格式要求明确逐题编号、不得遗漏。
"""
import importlib.util
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_PATH = PROJECT_ROOT / "UnipusAI_Helper.py"


def _stub_optional_modules():
    """为 v3.9.4 的可选 UI 依赖提供零依赖 stub（PyQt5/fluent_ui 不参与被测逻辑）。"""
    import types
    if "fluent_ui" not in sys.modules:
        stub = types.ModuleType("fluent_ui")

        class FluentModernGUI:
            def __init__(self, *args, **kwargs):
                pass

        stub.FluentModernGUI = FluentModernGUI
        sys.modules["fluent_ui"] = stub

def _load_main_module():
    _stub_optional_modules()
    spec = importlib.util.spec_from_file_location("unipus_helper_main_tests", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("unipus_helper_main_tests", module)
    spec.loader.exec_module(module)
    return module


mod = _load_main_module()
Question = mod.Question
QuestionType = mod.QuestionType
PromptBuilder = mod.PromptBuilder
GenericReplyStrategy = mod.GenericReplyStrategy


class _Client:
    accumulated_passages = []


class FakeNode:
    """最小化的 Selenium WebElement 桩：只支持本仓库用到的 find_element(s)/text。"""

    def __init__(self, text="", attrs=None, children=None):
        self._text = text
        self._attrs = attrs or {}
        self._children = children or {}

    @property
    def text(self):
        return self._text

    def get_attribute(self, name):
        return self._attrs.get(name)

    def is_displayed(self):
        return True

    def find_elements(self, by, value):
        return list(self._children.get(value, []))

    def find_element(self, by, value):
        found = self._children.get(value, [])
        if not found:
            raise Exception("no such element: " + value)
        return found[0]


class FakeDriver(FakeNode):
    pass


SUB_STEMS = [
    "What does the speech say about the institutional guarantee?",
    "How did local officials implement the pledges?",
    "What role did first Party secretaries play?",
    "Why were written pledges significant?",
    "How did the campaign affect rural grassroots organizations?",
    "What lasting mechanisms were established?",
]


def _call_build_text(builder, q, directions=""):
    """兼容 v3.9.4 _build_text(q) 与 v3.9.18 _build_text(q, directions)。"""
    import inspect
    if len(inspect.signature(builder._build_text).parameters) >= 3:
        return builder._build_text(q, directions)
    return builder._build_text(q)

def _make_group_container():
    textareas, boxes = [], []
    for i, stem in enumerate(SUB_STEMS, 1):
        area = FakeNode(attrs={"rows": "4"})
        header = FakeNode(text=f"{i}. {stem}")
        box = FakeNode(
            text=f"{i}. {stem}",
            children={"textarea": [area], ".question-inputbox-header": [header]},
        )
        textareas.append(area)
        boxes.append(box)
    container = FakeNode(
        text="material text for the group page",
        attrs={"class": "layout-container"},
        children={".question-inputbox": boxes, "textarea": list(textareas)},
    )
    return container, textareas


# ---------------------------------------------------------------- 修复点 1
def test_generic_reply_enumerates_sub_questions():
    container, textareas = _make_group_container()
    strategy = GenericReplyStrategy()
    q = strategy.parse(container, FakeDriver(), 1, "Answer the following questions.")
    assert q is not None and q.q_type == QuestionType.TEXT
    assert len(q.inputs) == 6
    assert "【小题列表】" in q.text
    for stem in SUB_STEMS:
        assert stem in q.text, "小题题干必须进入 prompt 上下文: " + stem
    assert "6 个小题" in q.text


def test_generic_reply_multi_textarea_without_headers_still_guides_numbering():
    areas = [FakeNode(), FakeNode()]
    container = FakeNode(
        text="short material",
        attrs={"class": "layout-container"},
        children={"textarea": list(areas)},
    )
    strategy = GenericReplyStrategy()
    q = strategy.parse(container, FakeDriver(), 1, "Answer the questions.")
    assert q is not None and q.q_type == QuestionType.TEXT
    assert len(q.inputs) == 2
    assert "【小题列表】" not in q.text
    assert "2 个输入框" in q.text or "2 个小题" in q.text


def test_generic_reply_single_textarea_keeps_legacy_shape():
    area = FakeNode()
    container = FakeNode(
        text="single answer box page",
        attrs={"class": "layout-container"},
        children={"textarea": [area]},
    )
    strategy = GenericReplyStrategy()
    q = strategy.parse(container, FakeDriver(), 1, "Write your answer.")
    assert q is not None and q.q_type == QuestionType.TEXT
    assert len(q.inputs) == 1
    assert "【小题列表】" not in q.text


# ---------------------------------------------------------------- 修复点 2
def test_build_text_lists_sub_question_stems():
    q = Question(
        number=1,
        text="【题目要求】Later in the same speech, President Xi summarized the campaign.",
        q_type=QuestionType.TEXT,
        element=None,
        inputs=[object() for _ in SUB_STEMS],
        banked_blanks=[{"index": i, "words": stem} for i, stem in enumerate(SUB_STEMS, 1)],
        directions="Answer the following questions.",
    )
    builder = PromptBuilder(_Client())
    lines = _call_build_text(builder, q, "Answer the following questions.")
    joined = "\n".join(lines)
    assert "【小题列表】" in joined
    for stem in SUB_STEMS:
        assert stem in joined
    assert "6 个小题" in joined


def test_build_text_without_blanks_still_requests_numbered_output():
    q = Question(
        number=1,
        text="【题目要求】Answer the questions.",
        q_type=QuestionType.TEXT,
        element=None,
        inputs=[object(), object()],
        directions="",
    )
    builder = PromptBuilder(_Client())
    joined = "\n".join(_call_build_text(builder, q))
    assert "2 个小题" in joined
    assert "按题号" in joined


def test_format_instructions_demand_numbered_short_answers():
    builder = PromptBuilder(_Client())
    joined = "\n".join(builder._format_instructions({QuestionType.TEXT: 2}))
    assert "简答题" in joined
    assert "逐题编号" in joined
    assert "每个小题都必须作答" in joined


TESTS = [
    test_generic_reply_enumerates_sub_questions,
    test_generic_reply_multi_textarea_without_headers_still_guides_numbering,
    test_generic_reply_single_textarea_keeps_legacy_shape,
    test_build_text_lists_sub_question_stems,
    test_build_text_without_blanks_still_requests_numbered_output,
    test_format_instructions_demand_numbered_short_answers,
]


def main():
    failures = []
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:
            failures.append(fn.__name__)
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"== {len(TESTS) - len(failures)}/{len(TESTS)} passed ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
