import json
import re
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
from transformers import AutoTokenizer

from utils import count_tokens

FULLWIDTH_BAR = "\uff5c"
SPECIAL_TOKEN_PATTERN = re.compile(
    r"(<\|[^>]+?\|>|<｜[^＞]+?｜>|</?think>|<\|im_start\|>|<\|im_end\|>|<s>|</s>)"
)


def get_default_results_path() -> Path:
    candidates = [
        Path(
            "results/gsm8k/gsm8k_deepseek-r1-distill-qwen-1.5b_20260209_152014.reparsed.json"
        ),
        Path(
            "results/gsm8k/gsm8k_deepseek-r1-distill-qwen-1.5b_20260209_152014.reparsed.json"
        ),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def discover_result_files() -> list[Path]:
    roots = [Path("results"), Path("results$")]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(root.rglob("*.json"))
    return sorted(set(files))


@st.cache_data(show_spinner=False)
def load_results(path_str: str) -> dict[str, Any]:
    with Path(path_str).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_chat_template(model_response: str) -> dict[str, str]:
    user_markers = [
        f"<{FULLWIDTH_BAR}User{FULLWIDTH_BAR}>",
        "<|User|>",
    ]
    assistant_markers = [
        f"<{FULLWIDTH_BAR}Assistant{FULLWIDTH_BAR}>",
        "<|Assistant|>",
    ]

    text = model_response or ""
    user_content = ""
    assistant_content = text

    for marker in user_markers:
        if marker in text:
            after_user = text.split(marker, 1)[1]
            break
    else:
        after_user = text

    for marker in assistant_markers:
        if marker in after_user:
            user_content, assistant_content = after_user.split(marker, 1)
            break

    return {
        "raw": text,
        "user_content": user_content.strip(),
        "assistant_content": assistant_content.strip(),
    }


def split_reasoning_and_final(assistant_content: str) -> tuple[str, str]:
    match = re.search(r"<think>\s*(.*?)\s*</think>", assistant_content, flags=re.DOTALL)
    if not match:
        return "", assistant_content.strip()
    reasoning = match.group(1).strip()
    final = re.sub(
        r"<think>.*?</think>", "", assistant_content, flags=re.DOTALL
    ).strip()
    return reasoning, final


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def escape_visible(text: str) -> str:
    """Render control characters as visible escape sequences (e.g., \\n)."""
    return (text or "").encode("unicode_escape").decode("ascii")


def render_raw_with_red_special_tokens(text: str) -> str:
    source = text or ""
    parts: list[str] = []
    last = 0
    for match in SPECIAL_TOKEN_PATTERN.finditer(source):
        start, end = match.span()
        parts.append(escape(source[last:start]))
        parts.append(
            "<span style='color:#ff4b4b;font-weight:700;'>"
            + escape(match.group(0))
            + "</span>"
        )
        last = end
    parts.append(escape(source[last:]))

    return (
        "<pre style='white-space: pre-wrap; word-break: break-word; margin:0; "
        "padding:0.75rem; border-radius:0.5rem; border:1px solid rgba(128,128,128,0.35); "
        "background:transparent; color:inherit;'>" + "".join(parts) + "</pre>"
    )


def count_special_tokens(text: str) -> int:
    return len(SPECIAL_TOKEN_PATTERN.findall(text or ""))


def has_unfinished_reasoning(model_response: str) -> bool:
    parsed = split_chat_template(model_response or "")
    assistant = parsed.get("assistant_content", "")
    open_count = assistant.count("<think>")
    close_count = assistant.count("</think>")
    return open_count > close_count


def compute_metrics(data: dict[str, Any]) -> dict[str, Any]:
    rows = data.get("results", [])
    total = len(rows)
    counted_correct = sum(1 for row in rows if row.get("is_correct") is True)
    unfinished = sum(
        1
        for row in rows
        if has_unfinished_reasoning(str(row.get("model_response", "")))
    )
    accuracy = (counted_correct / total) if total else 0.0
    return {
        "model": data.get("model", "unknown"),
        "total": total,
        "correct": counted_correct,
        "incorrect": total - counted_correct,
        "accuracy": accuracy,
        "unfinished_reasoning": unfinished,
        "unfinished_reasoning_pct": (unfinished / total) if total else 0.0,
    }


def row_matches_filter(row: dict[str, Any], only: str, query: str) -> bool:
    is_correct = bool(row.get("is_correct"))
    if only == "Correct only" and not is_correct:
        return False
    if only == "Incorrect only" and is_correct:
        return False
    if not query:
        return True
    haystack = " ".join(
        [
            str(row.get("question", "")),
            str(row.get("model_response", "")),
            str(row.get("ground_truth_solution", "")),
        ]
    ).lower()
    return query.lower() in haystack


def row_label(row_idx: int, row: dict[str, Any]) -> str:
    status = "OK" if bool(row.get("is_correct")) else "ERR"
    unfinished_tag = (
        " UNF" if has_unfinished_reasoning(str(row.get("model_response", ""))) else ""
    )
    question = str(row.get("question", "")).replace("\n", " ").strip()
    snippet = question[:95] + ("..." if len(question) > 95 else "")
    return f"#{row_idx:03d} [{status}{unfinished_tag}] {snippet}"


def main() -> None:
    st.set_page_config(page_title="GSM8K Evaluation Viewer", layout="wide")
    st.title("GSM8K Evaluation Viewer")

    files = discover_result_files()
    default_path = get_default_results_path()
    default_idx = files.index(default_path) if default_path in files else 0

    with st.sidebar:
        st.header("Run Selection")
        if files:
            selected_file = st.selectbox(
                "JSON results file",
                options=files,
                index=default_idx,
                format_func=lambda p: str(p),
            )
            path_str = str(selected_file)
        else:
            path_str = st.text_input("JSON results file", value=str(default_path))
        manual_path = st.text_input("Or type a path", value=path_str)
        show_raw = st.toggle(
            "Show raw model response",
            value=st.session_state.get("show_raw_model_response", False),
            key="show_raw_model_response",
        )
        render_markdown = st.toggle(
            "Render markdown",
            value=st.session_state.get("render_markdown", True),
            key="render_markdown",
        )

    path = Path(manual_path)
    if not path.exists():
        st.error(f"File not found: `{path}`")
        return

    data = load_results(str(path))
    metrics = compute_metrics(data)
    rows = data.get("results", [])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model", metrics["model"])
    c2.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c3.metric("Correct", metrics["correct"])
    c4.metric("Total", metrics["total"])
    c5.metric(
        "Unfinished <think>",
        f"{metrics['unfinished_reasoning_pct']:.2%}",
        f"{metrics['unfinished_reasoning']}/{metrics['total']}",
    )

    st.progress(metrics["accuracy"])
    st.caption(
        f"Incorrect: {metrics['incorrect']} | File accuracy field: "
        f"{data.get('accuracy', 'n/a')}"
    )

    st.divider()
    st.subheader("Examples")

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        filter_mode = st.selectbox(
            "Filter",
            options=["All", "Correct only", "Incorrect only"],
            index=0,
        )
    with search_col:
        search_query = st.text_input("Search text", value="")

    indexed_rows = list(enumerate(rows))
    filtered = [
        (idx, row)
        for idx, row in indexed_rows
        if row_matches_filter(row, filter_mode, search_query)
    ]
    if not filtered:
        st.warning("No examples match this filter.")
        return

    st.caption(f"Showing {len(filtered)} of {len(rows)} responses")

    if "nav_pos" not in st.session_state:
        st.session_state.nav_pos = 0
    st.session_state.nav_pos = min(int(st.session_state.nav_pos), len(filtered) - 1)

    nav1, nav2, nav3 = st.columns([1, 1, 4])
    with nav1:
        if st.button("Previous", use_container_width=True):
            st.session_state.nav_pos = max(0, int(st.session_state.nav_pos) - 1)
    with nav2:
        if st.button("Next", use_container_width=True):
            st.session_state.nav_pos = min(
                len(filtered) - 1, int(st.session_state.nav_pos) + 1
            )
    with nav3:
        pos_input = st.number_input(
            "Filtered position",
            min_value=0,
            max_value=len(filtered) - 1,
            value=int(st.session_state.nav_pos),
            step=1,
        )
        st.session_state.nav_pos = int(pos_input)

    option_ids = [idx for idx, _ in filtered]
    label_map = {idx: row_label(idx, row) for idx, row in filtered}
    current_pos = int(st.session_state.nav_pos)
    current_idx = option_ids[current_pos]

    selected_idx = st.selectbox(
        "Select response",
        options=option_ids,
        index=option_ids.index(current_idx),
        format_func=lambda i: label_map[i],
    )
    st.session_state.nav_pos = option_ids.index(selected_idx)

    row_idx, row = filtered[int(st.session_state.nav_pos)]
    st.caption(f"Original JSON index: {row_idx}")

    status = "Correct" if bool(row.get("is_correct")) else "Incorrect"
    st.markdown(f"**Status:** `{status}`")
    if has_unfinished_reasoning(str(row.get("model_response", ""))):
        st.error("Reasoning status: Incomplete (`<think>` block not closed)")
    else:
        st.success("Reasoning status: Complete")

    predicted = row.get("predicted_answer")
    gt_answer = row.get("ground_truth_answer")
    p1, p2 = st.columns(2)
    p1.metric("Predicted answer", str(predicted))
    p2.metric("Ground truth answer", str(gt_answer))

    parsed = split_chat_template(str(row.get("model_response", "")))
    reasoning, final_answer = split_reasoning_and_final(parsed["assistant_content"])

    tok = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True
    )
    reasoning_tokens = count_tokens(reasoning, tok)
    final_tokens = count_tokens(final_answer, tok)
    assistant_total_tokens = count_tokens(parsed["assistant_content"], tok)
    raw_total_tokens = count_tokens(parsed["raw"], tok)

    st.markdown("### Token Counts (Response)")
    t1, t2, t3, t4, t5 = st.columns(5)
    t1.metric("Reasoning", reasoning_tokens)
    t2.metric("Final", final_tokens)
    t3.metric("Assistant total", assistant_total_tokens)
    t4.metric("Raw total", raw_total_tokens)

    q_col, gt_col = st.columns(2)
    with q_col:
        st.markdown("### Prompt")
        question = str(row.get("question", ""))
        st.write(question)
        if parsed["user_content"] and normalize_text(
            parsed["user_content"]
        ) != normalize_text(question):
            st.markdown("### User Message in Chat Template")
            if render_markdown:
                st.markdown(parsed["user_content"])
            else:
                st.code(parsed["user_content"], language="text")
    with gt_col:
        st.markdown("### Ground Truth Solution")
        gt_solution = str(row.get("ground_truth_solution", ""))
        if render_markdown:
            st.markdown(gt_solution)
        else:
            st.code(gt_solution, language="text")

    r_col, a_col = st.columns(2)
    with r_col:
        st.markdown("### Reasoning Trace (`<think> ... </think>`)")
        if reasoning:
            if render_markdown:
                st.markdown(reasoning)
            else:
                st.code(reasoning, language="text")
        else:
            st.info("No explicit `<think>...</think>` block found.")
    with a_col:
        st.markdown("### Final Assistant Answer")
        if render_markdown:
            st.markdown(final_answer)
        else:
            st.code(final_answer, language="text")

    with st.expander("Raw Model Output", expanded=bool(show_raw)):
        if show_raw:
            st.markdown("### Raw Model Response")
            st.markdown(
                render_raw_with_red_special_tokens(parsed["raw"]),
                unsafe_allow_html=True,
            )

            st.markdown("### Raw Model Response (Escaped)")
            st.caption("Shows newline/control characters explicitly, e.g. `\\n`.")
            st.code(escape_visible(parsed["raw"]), language="text")
        else:
            st.caption("Enable `Show raw model response` in the sidebar.")


if __name__ == "__main__":
    main()
