"""User memory prompt configuration and search history sidebar panel."""

import json
from typing import Optional
import streamlit as st
from src.memory import UserMemoryStore


def render_memory_panel(memory_store: UserMemoryStore) -> None:
    """Renders the memory management controls and recent search history in the sidebar."""
    st.sidebar.markdown("### 🧠 Personalisation Memory")

    mem_data = memory_store.load_memory()
    user_context = mem_data.get("user_context", "")
    notes = mem_data.get("notes", "")
    preferred_topics = mem_data.get("preferred_topics", [])
    preferred_sectors = mem_data.get("preferred_sectors", [])

    with st.sidebar.expander("👤 User Profile & Memory Prompt", expanded=True):
        updated_context = st.text_area(
            "Legal Background / Interests",
            value=user_context,
            height=100,
            help="Your ongoing background context injected into prompts",
            key="mem_user_context_input",
        )

        updated_notes = st.text_area(
            "Analytical Directives",
            value=notes,
            height=80,
            help="Custom guidelines for summary synthesis",
            key="mem_notes_input",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Profile", use_container_width=True):
                mem_data["user_context"] = updated_context
                mem_data["notes"] = updated_notes
                memory_store.save_memory(mem_data)
                st.sidebar.success("Memory updated!")
        with col2:
            if st.button("🗑️ Reset", use_container_width=True):
                memory_store.clear_memory()
                st.sidebar.info("Memory reset to default.")
                st.rerun()

        # Download / export memory JSON
        st.download_button(
            label="📥 Export Memory JSON",
            data=json.dumps(mem_data, indent=2),
            file_name="prompt_context.json",
            mime="application/json",
            use_container_width=True,
        )

    # Search History Log
    with st.sidebar.expander("🕒 Recent Search History", expanded=False):
        history = memory_store.get_recent_searches(limit=8)
        if not history:
            st.write("No recorded searches yet.")
        else:
            for item in history:
                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <strong>{item['query']}</strong><br/>
                        <span style="color: #94a3b8; font-size: 0.75rem;">{item['timestamp'][:19]} · {item['model_used']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
