from __future__ import annotations

import hashlib
from collections import Counter
from types import SimpleNamespace

import pytest

from eval_harness.steering_data import (
    ROLES,
    _attack_rows,
    _balanced_templates,
    render_single_role,
)


class CharacterTokenizer:
    def __call__(self, text: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.mark.parametrize(
    ("role", "header"),
    [
        ("system", "system<|message|>"),
        ("user", "user<|message|>"),
        ("cot", "assistant<|channel|>analysis<|message|>"),
        ("assistant", "assistant<|channel|>final<|message|>"),
        (
            "tool",
            "functions. to=assistant<|channel|>commentary<|message|>",
        ),
    ],
)
def test_render_single_role_uses_native_gpt_oss_wrappers(
    role: str,
    header: str,
) -> None:
    content = "identical content"
    assert render_single_role(role, content) == (f"<|start|>{header}{content}<|end|>")


def test_render_single_role_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="unsupported role"):
        render_single_role("developer", "content")


def test_balanced_templates_is_deterministic_and_balanced_by_block() -> None:
    templates = [
        {"injection_id": f"template-{index}", "text": str(index)} for index in range(5)
    ]
    first = _balanced_templates(templates, count=30, seed=17)
    second = _balanced_templates(templates, count=30, seed=17)

    assert first == second
    assert first != _balanced_templates(templates, count=30, seed=18)
    for start in (0, 10, 20):
        assert set(
            Counter(row["injection_id"] for row in first[start : start + 10]).values()
        ) == {2}


def test_attack_rows_keep_clean_pairs_and_verified_injections() -> None:
    pages = []
    for index in range(30):
        html = f"<html><body>page {index}</body></html>"
        pages.append(
            {
                "source_url": f"https://example.test/wiki/{index}",
                "html": html,
                "html_sha256": hashlib.sha256(html.encode()).hexdigest(),
                "content_sha256": hashlib.sha256(f"page {index}".encode()).hexdigest(),
                "html_bytes": len(html.encode()),
            }
        )
    templates = {
        family: [
            {
                "injection_id": f"{family}:template-{index}",
                "text": f'<p data-family="{family}">instruction {index}</p>',
            }
            for index in range(5)
        ]
        for family in ("base-injection", "cot-forgery-injection")
    }

    outputs = _attack_rows(pages, templates, CharacterTokenizer(), seed=101)

    assert set(outputs) == {"layer", "tune", "devval"}
    assert all(len(rows) == 30 for rows in outputs.values())
    for rows in outputs.values():
        assert Counter(row["variant"] for row in rows) == {
            "clean": 10,
            "base-injection": 10,
            "cot-forgery-injection": 10,
        }
        assert set(
            Counter(
                row["injection_id"] for row in rows if row["variant"] != "clean"
            ).values()
        ) == {2}
        for page_id in {row["page_id"] for row in rows}:
            paired = [row for row in rows if row["page_id"] == page_id]
            clean = next(row for row in paired if row["variant"] == "clean")
            assert all(row["clean_case_id"] == clean["case_id"] for row in paired)
            assert clean["html_sha256"] == clean["clean_html_sha256"]
            for attack in (row for row in paired if row["variant"] != "clean"):
                assert attack["injection_context_verified"] is True
                assert attack["injection_sha256"] is not None
                assert attack["html_sha256"] != clean["html_sha256"]
                assert attack["html"].endswith("</body></html>")


def test_role_set_stays_frozen() -> None:
    assert ROLES == ("system", "user", "cot", "assistant", "tool")
