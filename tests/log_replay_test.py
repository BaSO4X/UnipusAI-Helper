# -*- coding: utf-8 -*-
"""端到端回放 2026-11-27 10:50 日志场景：6 小题简答组。

验证链路：GenericReplyStrategy 解析 → PromptBuilder 生成 prompt →
AI 按新契约返回编号答案 → 现有 _parse_banked_answer 提取 6/6。
"""
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _stub_optional_modules():
    import types
    if "fluent_ui" not in sys.modules:
        stub = types.ModuleType("fluent_ui")
        class FluentModernGUI:
            def __init__(self, *args, **kwargs):
                pass
        stub.FluentModernGUI = FluentModernGUI
        sys.modules["fluent_ui"] = stub

_stub_optional_modules()

spec = importlib.util.spec_from_file_location(
    "unipus_helper_main_replay", PROJECT_ROOT / "UnipusAI_Helper.py")
mod = importlib.util.module_from_spec(spec)
sys.modules.setdefault("unipus_helper_main_replay", mod)
spec.loader.exec_module(mod)

GenericReplyStrategy = mod.GenericReplyStrategy
PromptBuilder = mod.PromptBuilder


class FakeNode:
    def __init__(self, text="", attrs=None, children=None):
        self._text, self._attrs, self._children = text, attrs or {}, children or {}

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


LOG_STEMS = [
    "What does the speech say about the institutional guarantee for poverty alleviation?",
    "How did local officials implement the pledges?",
    "What role did first Party secretaries play in the campaign?",
    "Why were the written pledges significant?",
    "How did the campaign affect rural grassroots Party organizations?",
    "What lasting mechanisms were established?",
]
DIRECTION = ("【题目要求】Later in the same speech, President Xi summarized the campaign. "
             "Answer the following questions.")


def _container():
    textareas, boxes = [], []
    for i, stem in enumerate(LOG_STEMS, 1):
        area = FakeNode(attrs={"rows": "4"})
        boxes.append(FakeNode(
            text=f"{i}. {stem}",
            children={"textarea": [area],
                      ".question-inputbox-header": [FakeNode(text=f"{i}. {stem}")]},
        ))
        textareas.append(area)
    return FakeNode(
        text="page material about the speech",
        attrs={"class": "layout-container"},
        children={".question-inputbox": boxes, "textarea": list(textareas)},
    )


class _Client:
    accumulated_passages = []


def main():
    # 1) 解析：新版 GenericReplyStrategy
    q = GenericReplyStrategy().parse(_container(), FakeNode(), 1,
                                     "Answer the following questions.")
    assert q is not None and q.q_type == mod.QuestionType.TEXT, "解析失败"
    assert len(q.inputs) == 6, f"应有 6 个输入框，实际 {len(q.inputs)}"

    # 2) prompt：应包含题目要求 + 全部 6 个小题题干 + 编号要求
    prompt = PromptBuilder(_Client()).build([q], "")
    for stem in LOG_STEMS:
        assert stem in prompt, "小题题干缺失: " + stem[:40]
    assert "逐题" in prompt or "按题号" in prompt
    print("--- 生成的 prompt（截取）---")
    print("\n".join(prompt.splitlines()[:12]))

    # 3) AI 按新契约返回编号答案 → 现有提取器应 6/6 全提出来
    numbered_response = (
        "简答题: "
        "1. The speech shows that the CPC provided the top-level leadership and built a working "
        "mechanism whereby the central leadership makes overall plans and provincial authorities "
        "take overall responsibility, giving an institutional guarantee.\n"
        "2. Local officials signed written pledges so that goals were clear and accountability "
        "was ensured at every level.\n"
        "3. First Party secretaries were dispatched to impoverished villages to strengthen "
        "grassroots leadership on the ground.\n"
        "4. The pledges turned abstract targets into concrete, accountable commitments.\n"
        "5. It enhanced the cohesion and vitality of rural grassroots Party organizations.\n"
        "6. A lasting mechanism of targeted assistance with paired-up support was established."
    )
    answers = mod.AnswerExecutor._parse_banked_answer(numbered_response, 6)
    filled = [a for a in answers if a.strip()]
    assert len(filled) == 6, f"应提取 6 个答案，实际 {len(filled)}: {answers}"
    for i, a in enumerate(answers, 1):
        print(f"空{i}: {a[:60]}...")

    # 4) 兼容旧行为：AI 仍返回整段 blob 时，提取不抛异常（由修复后的重试循环兜底）
    blob = "简答题: " + "A long single paragraph answer. " * 30
    legacy = mod.AnswerExecutor._parse_banked_answer(blob, 6)
    assert isinstance(legacy, list) and len(legacy) == 6

    print("== 回放通过：解析/prompt/提取 全链路 OK ==")


if __name__ == "__main__":
    main()
