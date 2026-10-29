"""Results display component with faceted filtering, pagination, similar document actions, and Case 3 refusal notices (Section 4.7)."""

import math
from typing import Any, Callable, Dict, List, Optional
import streamlit as st


def render_summary(summary_data: Dict[str, Any]) -> None:
    """Renders the synthesised legal summary with model and personalisation badges."""
    summary_text = summary_data.get("summary", "")
    model_used = summary_data.get("model_used", "Local LLM")
    status = summary_data.get("status", "success")
    docs_analyzed = summary_data.get("docs_analyzed", 0)
    memory_applied = summary_data.get("memory_applied", False)

    docs_badge = (
        f'<span class="badge badge-score" style="margin-left: auto;">Synthesised: {docs_analyzed} Acts</span>'
        if docs_analyzed
        else ""
    )
    mem_badge = (
        '<span class="badge" style="background: rgba(139,92,246,0.25); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.5);">🧠 Personalised</span>'
        if memory_applied
        else ""
    )

    st.markdown(
        f"""
        <div class="summary-container">
            <div class="summary-title">
                <span>📑 Executive Legal Analysis</span>
                <div style="margin-left: auto; display: flex; gap: 0.5rem; align-items: center;">
                    {mem_badge}
                    {docs_badge}
                    <span class="badge badge-celex">Model: {model_used}</span>
                </div>
            </div>

{summary_text}

        </div>
        """,
        unsafe_allow_html=True,
    )


def render_articles(
    results: List[Dict[str, Any]],
    on_similar_click: Optional[Callable[[str], None]] = None,
) -> None:
    """Renders faceted filtering, pagination, and expandable document cards."""
    if not results:
        st.info("No matching legislative documents retrieved.")
        return

    # Check for Case 3 Zero-Match condition across top matches
    top_doc = results[0]
    is_zero_match = top_doc.get("is_below_null_threshold", False)
    threshold = top_doc.get("null_threshold", 0.50)

    if is_zero_match:
        peak_score = top_doc.get("peak_dense_score", top_doc.get("peak_score", 0.0))
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1.2rem;">
                <div style="font-weight: 600; color: #fca5a5; display: flex; align-items: center; gap: 0.5rem;">
                    ⚠️ Graceful Refusal Triggered (Case 3 Zero-Match)
                </div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 0.4rem;">
                    Top candidate similarity (<strong>{int(peak_score * 100)}%</strong>) falls below the calibrated statutory null cutoff (<strong>{int(threshold * 100)}%</strong>).
                    The indexed EUR-Lex database does not contain acts regulating this query. The documents displayed below are background legal distractors.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 1. Faceted Filters Expander
    with st.expander("⚙️ Faceted Results Filtering & Sorting", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)

        doc_types = sorted(list(dict.fromkeys(d.get("type", "Act") for d in results)))
        with col_f1:
            selected_types = st.multiselect(
                "Document Types",
                options=doc_types,
                default=doc_types,
                key="filter_doc_types",
            )

        with col_f2:
            min_score = st.slider(
                "Min Similarity %",
                min_value=0,
                max_value=90,
                value=0,
                step=5,
                key="filter_min_score",
            )

        with col_f3:
            sort_by = st.selectbox(
                "Sort Results By",
                options=["Relevance Score (Descending)", "Date (Newest First)", "Date (Oldest First)"],
                index=0,
                key="sort_results_by",
            )

    # Apply Filters
    filtered = [
        d for d in results
        if d.get("type", "Act") in selected_types
        and int(d.get("peak_score", 0.0) * 100) >= min_score
    ]

    # Apply Sorting
    if "Date (Newest First)" in sort_by:
        filtered.sort(key=lambda d: d.get("date", "0000-00-00"), reverse=True)
    elif "Date (Oldest First)" in sort_by:
        filtered.sort(key=lambda d: d.get("date", "9999-99-99"))
    else:
        filtered.sort(key=lambda d: d.get("peak_score", 0.0), reverse=True)

    total_filtered = len(filtered)
    st.subheader(f"🏛️ Retrieved Legislation & Case Articles ({total_filtered} matches)")

    if total_filtered == 0:
        st.warning("No documents match the current active filters. Adjust your filter criteria above.")
        return

    # 2. Pagination Controls
    col_p1, col_p2 = st.columns([3, 1])
    with col_p2:
        page_size = st.selectbox("Results per page", options=[5, 10, 20], index=0, key="page_size_selector")

    total_pages = max(1, math.ceil(total_filtered / page_size))
    current_page = st.session_state.get("current_results_page", 1)
    if current_page > total_pages:
        current_page = 1
        st.session_state["current_results_page"] = 1

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_filtered)
    page_docs = filtered[start_idx:end_idx]

    # 3. Document Cards Display
    for display_num, doc in enumerate(page_docs, start=start_idx + 1):
        celex = doc.get("celex", "N/A")
        title = doc.get("title", "Untitled Document")
        date = doc.get("date", "N/A")
        doc_type = doc.get("type", "Act")
        peak_score = doc.get("peak_score", 0.0)
        dense_score = doc.get("peak_dense_score", peak_score)
        excerpt = doc.get("best_chunk", "")
        tags = doc.get("tags", {})

        jurisdiction_tags = [t["category"] for t in tags.get("Jurisdiction", [])]
        topic_tags = [t["category"] for t in tags.get("Regulatory Topics", [])]
        sector_tags = [t["category"] for t in tags.get("Industry / Sector", [])]

        score_pct = int(peak_score * 100)
        expander_title = f"#{display_num} · {title} [{celex}] · Match {score_pct}%"

        with st.expander(expander_title, expanded=(display_num == 1)):
            badges_html = [
                f'<span class="badge badge-celex">CELEX {celex}</span>',
                f'<span class="badge badge-score">Combined: {score_pct}%</span>',
                f'<span class="badge" style="background: rgba(59,130,246,0.2); color: #93c5fd;">Dense: {int(dense_score * 100)}%</span>',
                f'<span class="badge" style="background: rgba(255,255,255,0.1); color: #e2e8f0;">📅 {date}</span>',
                f'<span class="badge" style="background: rgba(255,255,255,0.1); color: #e2e8f0;">📂 {doc_type}</span>',
            ]

            for j in jurisdiction_tags:
                badges_html.append(f'<span class="badge badge-jurisdiction">🌍 {j}</span>')
            for t in topic_tags:
                badges_html.append(f'<span class="badge badge-topic">⚖️ {t}</span>')
            for s in sector_tags:
                badges_html.append(f'<span class="badge badge-sector">🏢 {s}</span>')

            st.markdown(f"<div style='margin-bottom: 0.5rem;'>{''.join(badges_html)}</div>", unsafe_allow_html=True)

            st.markdown("#### Key Relevant Meso-Chunk Excerpt (~2,000 tokens)")
            st.info(excerpt)

            # Option to view full text
            full_text = doc.get("full_text_excerpt", "")
            if full_text and full_text != excerpt:
                with st.expander("Show Extended Context"):
                    st.text(full_text)

            # Similar Document Button
            if st.button(f"🔗 Find Legislation Similar to CELEX {celex}", key=f"btn_sim_{celex}_{display_num}"):
                if on_similar_click:
                    on_similar_click(celex)
                else:
                    st.session_state["execute_similar_celex"] = celex
                    st.rerun()

    # Pagination navigation controls
    if total_pages > 1:
        st.divider()
        c_prev, c_info, c_next = st.columns([1, 2, 1])
        with c_prev:
            if st.button("⬅️ Previous Page", disabled=(current_page <= 1), key="btn_page_prev"):
                st.session_state["current_results_page"] = current_page - 1
                st.rerun()
        with c_info:
            st.markdown(
                f"<div style='text-align: center; color: #94a3b8; font-size: 0.9rem; padding-top: 0.4rem;'>"
                f"Page <strong>{current_page}</strong> of <strong>{total_pages}</strong> (Showing {start_idx + 1}–{end_idx} of {total_filtered})"
                f"</div>",
                unsafe_allow_html=True,
            )
        with c_next:
            if st.button("Next Page ➡️", disabled=(current_page >= total_pages), key="btn_page_next"):
                st.session_state["current_results_page"] = current_page + 1
                st.rerun()
