"""Results display component for synthesised summaries and legislative articles."""

from typing import Any, Dict, List
import streamlit as st


def render_summary(summary_data: Dict[str, Any]) -> None:
    """Renders the Ollama synthesised legal summary with badges."""
    summary_text = summary_data.get("summary", "")
    model_used = summary_data.get("model_used", "Local Ollama")
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


def render_articles(results: List[Dict[str, Any]]) -> None:
    """Renders expandable document cards with metadata badges and excerpts."""
    if not results:
        st.info("No matching legislative documents retrieved.")
        return

    st.subheader(f"🏛️ Retrieved Legislation & Case Articles ({len(results)} matches)")

    for idx, doc in enumerate(results, 1):
        celex = doc.get("celex", "N/A")
        title = doc.get("title", "Untitled Document")
        date = doc.get("date", "N/A")
        doc_type = doc.get("type", "Act")
        peak_score = doc.get("peak_score", 0.0)
        excerpt = doc.get("best_chunk", "")
        tags = doc.get("tags", {})

        jurisdiction_tags = [t["category"] for t in tags.get("Jurisdiction", [])]
        topic_tags = [t["category"] for t in tags.get("Regulatory Topics", [])]
        sector_tags = [t["category"] for t in tags.get("Industry / Sector", [])]

        score_pct = int(peak_score * 100)
        expander_title = f"#{idx} · {title} [{celex}] · Match {score_pct}%"

        with st.expander(expander_title, expanded=(idx == 1)):
            # Metadata Badge row
            badges_html = [
                f'<span class="badge badge-celex">CELEX {celex}</span>',
                f'<span class="badge badge-score">Relevance: {score_pct}%</span>',
                f'<span class="badge" style="background: rgba(255,255,255,0.1); color: #e2e8f0;">📅 {date}</span>',
                f'<span class="badge" style="background: rgba(255,255,255,0.1); color: #e2e8f0;">📂 {doc_type}</span>',
            ]

            for j in jurisdiction_tags:
                badges_html.append(f'<span class="badge badge-jurisdiction">🌍 {j}</span>')
            for t in topic_tags:
                badges_html.append(f'<span class="badge badge-topic">⚖️ {t}</span>')
            for s in sector_tags:
                badges_html.append(f'<span class="badge badge-sector">🏢 {s}</span>')

            st.markdown(f"<div>{''.join(badges_html)}</div>", unsafe_allow_html=True)

            st.markdown("#### Key Relevant Excerpt")
            st.info(excerpt)

            # Option to view full text
            full_text = doc.get("full_text_excerpt", "")
            if full_text and full_text != excerpt:
                with st.expander("Show Extended Context"):
                    st.text(full_text)
