"""Search bar and query configuration component."""

from typing import Dict, Tuple
import streamlit as st
import config


def render_search_bar() -> Tuple[str, bool, str, int, bool]:
    """Renders the search query input, model selector, memory toggle, and action button.
    
    Returns:
        (query, enable_memory, selected_model, top_k, submitted)
    """
    col1, col2 = st.columns([4, 1])

    with col1:
        query = st.text_input(
            label="Search EUR-Lex Legislation",
            value="",
            placeholder="e.g. What are the requirements and penalties for high-risk AI deployers?",
            label_visibility="collapsed",
            key="main_search_query_input",
        )

    with col2:
        submitted = st.button("🔍 Search", use_container_width=True, type="primary")

    # Configuration row
    c1, c2, c3 = st.columns([1.5, 1.5, 1.0])

    with c1:
        enable_memory = st.checkbox(
            "🧠 Enable Personalised Memory Prompt",
            value=True,
            help="Inject your saved background profile, preferred domains, and directives into retrieval & summarisation",
            key="enable_memory_toggle",
        )

    with c2:
        model_keys = list(config.OLLAMA_MODELS.keys())
        selected_model = st.selectbox(
            "LLM Summariser",
            options=model_keys,
            format_func=lambda k: config.OLLAMA_MODELS[k],
            index=0,
            help="Select local Ollama model for legal synthesis (runs on RTX A2000)",
            key="selected_ollama_model",
        )

    with c3:
        top_k = st.slider(
            "Max Articles",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Maximum retrieved legislative articles to include in analysis",
            key="top_k_slider",
        )

    return query, enable_memory, selected_model, top_k, submitted
