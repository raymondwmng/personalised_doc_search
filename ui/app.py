"""Main Streamlit Application for Personalised EUR-Lex Document Search POC."""

import logging
from pathlib import Path
import sqlite3
import sys

# Ensure repository root is in sys.path regardless of launch cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

import config
from src.memory import UserMemoryStore
from src.searcher import SearchEngine
from src.summariser import LegalSummariser
from ui.components.charts import render_analytics_dashboard
from ui.components.memory_panel import render_memory_panel
from ui.components.results_display import render_articles, render_summary
from ui.components.search_bar import render_search_bar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Personalised EUR-Lex Legal Search",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_services():
    """Initialises search engine, summariser, and memory store once."""
    mem_store = UserMemoryStore()
    search_engine = SearchEngine(memory_store=mem_store)
    summariser = LegalSummariser()
    return mem_store, search_engine, summariser


def get_db_stats() -> dict:
    """Fetches total document and chunk counts from SQLite."""
    if not config.DOCUMENTS_DB_PATH.exists():
        return {"docs": 0, "chunks": 0}
    try:
        conn = sqlite3.connect(str(config.DOCUMENTS_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        docs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM chunks")
        chunks = cursor.fetchone()[0]
        conn.close()
        return {"docs": docs, "chunks": chunks}
    except Exception:
        return {"docs": 0, "chunks": 0}


def main():
    memory_store, search_engine, summariser = get_services()
    db_stats = get_db_stats()

    # Sidebar: Memory & History Panel
    render_memory_panel(memory_store)

    # Hero Banner
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-title">⚖️ Personalised EUR-Lex Legal Search</div>
            <div class="hero-subtitle">
                Dense semantic retrieval over EU legislation, multi-dimensional taxonomy profiling (Jurisdiction, Topics, Sectors),
                and local LLM synthesis powered by RTX A2000 GPU acceleration.
            </div>
            <div style="margin-top: 0.8rem; display: flex; gap: 0.75rem; flex-wrap: wrap;">
                <span class="badge" style="background: rgba(99,102,241,0.2); color: #a5b4fc;">📚 Indexed Docs: {db_stats['docs']}</span>
                <span class="badge" style="background: rgba(16,185,129,0.2); color: #6ee7b7;">🧩 Vector Chunks: {db_stats['chunks']}</span>
                <span class="badge" style="background: rgba(6,182,212,0.2); color: #67e8f9;">⚡ GPU: RTX A2000 8GB</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Search Bar
    query, enable_memory, selected_model, top_k, submitted = render_search_bar()

    # Active Memory Status Strip
    if enable_memory:
        mem_profile = memory_store.load_memory()
        topics = mem_profile.get("preferred_topics", [])
        sectors = mem_profile.get("preferred_sectors", [])
        context = mem_profile.get("user_context", "")
        notes = mem_profile.get("notes", "")

        active_badges = []
        if context:
            active_badges.append(f'<span class="badge" style="background: rgba(99,102,241,0.2); color: #c7d2fe;">👤 Background: {len(context)} chars</span>')
        for t in topics:
            active_badges.append(f'<span class="badge badge-topic">⚖️ {t}</span>')
        for s in sectors:
            active_badges.append(f'<span class="badge badge-sector">🏢 {s}</span>')
        if notes:
            active_badges.append(f'<span class="badge" style="background: rgba(16,185,129,0.2); color: #a7f3d0;">🎯 Directives: {len(notes)} chars</span>')

        with st.expander(f"🧠 Active Personalised Memory Prompt ({len(topics)} Topics · {len(sectors)} Sectors · Background {'✓' if context else '✗'} · Directives {'✓' if notes else '✗'})", expanded=False):
            if active_badges:
                st.markdown(f"<div style='margin-bottom: 0.5rem;'>{''.join(active_badges)}</div>", unsafe_allow_html=True)
            prompt_str = memory_store.get_prompt_context_string()
            if prompt_str:
                st.code(prompt_str, language="text")
            else:
                st.info("Memory is enabled but all 4 fields are currently blank. Customise them in the sidebar.")

    # Execute Search
    if submitted and query.strip():
        with st.spinner("🔍 Performing semantic retrieval & hybrid re-ranking..."):
            try:
                results = search_engine.search(
                    query=query.strip(),
                    top_k_chunks=top_k * 4,
                    top_n_docs=top_k,
                    enable_memory=enable_memory,
                )
                st.session_state["search_results"] = results
                st.session_state["last_query"] = query
            except Exception as e:
                st.error(f"Search retrieval error: {e}")
                logger.error(f"Search error: {e}", exc_info=True)
                return

        if results:
            memory_ctx = memory_store.get_prompt_context_string() if enable_memory else None
            with st.spinner(f"🧠 Synthesising legal summary with local model '{selected_model}'..."):
                summary_data = summariser.summarise(
                    query=query.strip(),
                    retrieved_docs=results,
                    model_name=selected_model,
                    memory_context=memory_ctx,
                    max_docs=top_k,
                )
                summary_data["memory_applied"] = bool(memory_ctx)
                st.session_state["summary_data"] = summary_data

                # Log to persistent history
                top_celex = [d.get("celex", "") for d in results[:top_k]]
                memory_store.log_search(
                    query=query.strip(),
                    top_celex_list=top_celex,
                    summary=summary_data.get("summary", ""),
                    model_used=selected_model,
                )
        else:
            st.warning("No matching legislative documents or articles found for this query.")
            st.session_state["search_results"] = []
            st.session_state["summary_data"] = None

    # Render persisted results from session state
    if "search_results" in st.session_state and st.session_state["search_results"]:
        results = st.session_state["search_results"]
        summary_data = st.session_state.get("summary_data")

        # 1. Summary Box
        if summary_data:
            render_summary(summary_data)

        # 2. Analytics Dashboard (Radar Chart & Histogram)
        render_analytics_dashboard(results)

        # 3. Retrieved Articles
        st.divider()
        render_articles(results)


if __name__ == "__main__":
    main()
