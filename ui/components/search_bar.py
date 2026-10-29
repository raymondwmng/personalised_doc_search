"""Search bar and query configuration component supporting dual embedding models and LLM summarisers."""

from typing import Dict, Tuple
import streamlit as st
import config


def render_search_bar() -> Tuple[str, bool, str, str, int, bool]:
    """Renders the search query input, embedding model selector, LLM selector, memory toggle, and action button.
    
    Returns:
        (query, enable_memory, selected_embed_model, selected_llm_model, top_k, submitted)
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
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 0.9])

    with c1:
        enable_memory = st.checkbox(
            "🧠 Enable Memory Prompt",
            value=True,
            help="Inject your saved background profile, preferred domains, and directives into retrieval & summarisation",
            key="enable_memory_toggle",
        )

    with c2:
        embed_keys = list(config.SUPPORTED_EMBEDDING_MODELS.keys())
        default_embed_idx = embed_keys.index(config.DEFAULT_EMBEDDING_MODEL) if config.DEFAULT_EMBEDDING_MODEL in embed_keys else 0
        selected_embed_model = st.selectbox(
            "Embedding Backbone",
            options=embed_keys,
            format_func=lambda k: config.SUPPORTED_EMBEDDING_MODELS[k]["display_name"],
            index=default_embed_idx,
            help="Select BAAI/bge-m3 (SOTA multilingual EU law) or ModernBERT-base (high-throughput)",
            key="selected_embedding_model_key",
        )

    with c3:
        model_keys = list(config.ALL_MODELS.keys())
        default_idx = model_keys.index(config.DEFAULT_ANALYTICAL_MODEL) if config.DEFAULT_ANALYTICAL_MODEL in model_keys else 0
        selected_llm_model = st.selectbox(
            "LLM Summariser",
            options=model_keys,
            format_func=lambda k: config.ALL_MODELS[k],
            index=default_idx,
            help="Select Mistral API (cloud) or local Ollama model for legal synthesis",
            key="selected_llm_model",
        )

    with c4:
        top_k = st.slider(
            "Max Articles",
            min_value=3,
            max_value=20,
            value=5,
            step=1,
            help="Maximum retrieved legislative articles to include in analysis",
            key="top_k_slider",
        )

    return query, enable_memory, selected_embed_model, selected_llm_model, top_k, submitted
