"""Main Streamlit Application for Personalised EUR-Lex Document Search POC (Stage 2)."""

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
def get_services(embedding_model_key: str = config.DEFAULT_EMBEDDING_MODEL):
    """Initialises search engine, summariser, and memory store once per model key."""
    mem_store = UserMemoryStore()
    search_engine = SearchEngine(model_name_or_key=embedding_model_key, memory_store=mem_store)
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
    # Search Bar & Controls
    query, enable_memory, selected_embed_model, selected_llm_model, top_k, submitted = render_search_bar()

    # Initialise backend services for selected embedding backbone
    memory_store, search_engine, summariser = get_services(selected_embed_model)
    if search_engine.model_key != selected_embed_model:
        search_engine.switch_model(selected_embed_model)

    db_stats = get_db_stats()
    embed_spec = config.resolve_embedding_spec(selected_embed_model)

    # Sidebar: Memory & History Panel
    render_memory_panel(memory_store)

    # Hero Banner
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-title">⚖️ Personalised EUR-Lex Legal Search</div>
            <div class="hero-subtitle">
                Dense semantic retrieval over EU legislation with <strong>Meso (~2,000 tokens)</strong> chapter chunking,
                hybrid BM25Okapi fusion, multi-dimensional taxonomy profiling, and GPU-accelerated local LLM synthesis.
            </div>
            <div style="margin-top: 0.8rem; display: flex; gap: 0.75rem; flex-wrap: wrap;">
                <span class="badge" style="background: rgba(99,102,241,0.2); color: #a5b4fc;">📚 Indexed Docs: {db_stats['docs']}</span>
                <span class="badge" style="background: rgba(16,185,129,0.2); color: #6ee7b7;">🧩 Meso Chunks (~2000t): {db_stats['chunks']}</span>
                <span class="badge" style="background: rgba(245,158,11,0.2); color: #fcd34d;">🔍 Model: {embed_spec['display_name']}</span>
                <span class="badge" style="background: rgba(6,182,212,0.2); color: #67e8f9;">⚡ GPU: RTX A2000 8GB</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check for "Find Similar Legislation" trigger from result cards
    similar_celex = st.session_state.pop("execute_similar_celex", None)
    if similar_celex:
        with st.spinner(f"🔍 Finding legislative acts semantically related to CELEX {similar_celex}..."):
            try:
                results = search_engine.search_similar_documents(celex=similar_celex, top_n_docs=top_k)
                st.session_state["search_results"] = results
                st.session_state["last_query"] = f"Similar to CELEX {similar_celex}"
                st.session_state["summary_data"] = None
                st.session_state["current_results_page"] = 1
            except Exception as e:
                st.error(f"Similar document search error: {e}")

    # Active Memory Status Strip
    if enable_memory:
        mem_profile = memory_store.load_memory()
        topics = mem_profile.get("preferred_topics", [])
        sectors = mem_profile.get("preferred_sectors", [])
        context = mem_profile.get("user_context", "")
        notes = mem_profile.get("notes", "")

        active_badges = []
        if context:
            active_badges.append(f'<span class="badge" style="background: rgba(99,102,241,0.2); color: #c7d2fe;">👤 Persona: {len(context)} chars</span>')
        for t in topics:
            active_badges.append(f'<span class="badge badge-topic">⚖️ {t}</span>')
        for s in sectors:
            active_badges.append(f'<span class="badge badge-sector">🏢 {s}</span>')
        if notes:
            active_badges.append(f'<span class="badge" style="background: rgba(16,185,129,0.2); color: #a7f3d0;">🎯 Directives: {len(notes)} chars</span>')

        with st.expander(f"🧠 Active Personalised Memory Prompt ({len(topics)} Topics · {len(sectors)} Sectors · Persona {'✓' if context else '✗'} · Directives {'✓' if notes else '✗'})", expanded=False):
            if active_badges:
                st.markdown(f"<div style='margin-bottom: 0.5rem;'>{''.join(active_badges)}</div>", unsafe_allow_html=True)
            prompt_str = memory_store.get_prompt_context_string()
            if prompt_str:
                st.code(prompt_str, language="text")
            else:
                st.info("Memory is enabled but all 4 fields are currently blank. Customise them in the sidebar.")

    # Execute Search
    if submitted and query.strip():
        with st.spinner(f"🔍 Searching with {embed_spec['display_name']} + BM25Okapi hybrid re-ranking..."):
            try:
                results = search_engine.search(
                    query=query.strip(),
                    top_k_chunks=top_k * 4,
                    top_n_docs=top_k,
                    enable_memory=enable_memory,
                )
                st.session_state["search_results"] = results
                st.session_state["last_query"] = query
                st.session_state["current_results_page"] = 1
            except Exception as e:
                st.error(f"Search retrieval error: {e}")
                logger.error(f"Search error: {e}", exc_info=True)
                return

        if results:
            memory_ctx = memory_store.get_prompt_context_string() if enable_memory else None
            model_type_str = "Mistral Cloud API" if config.is_mistral_model(selected_llm_model) else "local Ollama"
            model_display_name = config.ALL_MODELS.get(selected_llm_model, selected_llm_model)
            with st.spinner(f"🧠 Synthesising legal briefing with {model_type_str} '{model_display_name}'..."):
                summary_data = summariser.summarise(
                    query=query.strip(),
                    retrieved_docs=results,
                    model_name=selected_llm_model,
                    memory_context=memory_ctx,
                    max_docs=top_k,
                )
                summary_data["memory_applied"] = bool(memory_ctx)
                st.session_state["summary_data"] = summary_data

                # Accumulate interaction into persistent memory
                memory_store.accumulate_interaction(
                    query=query.strip(),
                    retrieved_docs=results,
                    summary=summary_data.get("summary", ""),
                    model_used=selected_llm_model,
                )

                # Log to persistent history DB
                top_celex = [d.get("celex", "") for d in results[:top_k]]
                memory_store.log_search(
                    query=query.strip(),
                    top_celex_list=top_celex,
                    summary=summary_data.get("summary", ""),
                    model_used=selected_llm_model,
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

        # 3. Retrieved Articles with Filters and Pagination
        st.divider()
        render_articles(results)


if __name__ == "__main__":
    main()
