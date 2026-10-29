"""User memory prompt configuration and search history sidebar panel."""

import json
from datetime import datetime
from typing import List, Optional
import streamlit as st

import config
from src.memory import DEFAULT_MEMORY, UserMemoryStore


def _cb_save_profile(memory_store: UserMemoryStore) -> None:
    """Callback triggered before widget instantiation to save memory profile."""
    new_profile = {
        "user_context": st.session_state.get("mem_user_context_input", "").strip(),
        "preferred_topics": list(st.session_state.get("mem_topics_input", [])),
        "preferred_sectors": list(st.session_state.get("mem_sectors_input", [])),
        "notes": st.session_state.get("mem_notes_input", "").strip(),
    }
    memory_store.save_memory(new_profile)
    st.session_state["mem_status_notification"] = ("success", "✅ Memory profile updated successfully!")


def _cb_reset_defaults(memory_store: UserMemoryStore) -> None:
    """Callback triggered before widget instantiation to reset profile to defaults."""
    default_state = memory_store.reset_to_defaults()
    st.session_state["mem_user_context_input"] = default_state.get("user_context", "")
    st.session_state["mem_topics_input"] = list(default_state.get("preferred_topics", []))
    st.session_state["mem_sectors_input"] = list(default_state.get("preferred_sectors", []))
    st.session_state["mem_notes_input"] = default_state.get("notes", "")
    st.session_state["mem_status_notification"] = ("info", "🔄 Memory reset to default template.")


def _cb_clear_all(memory_store: UserMemoryStore) -> None:
    """Callback triggered before widget instantiation to clear all profile fields."""
    memory_store.clear_memory()
    st.session_state["mem_user_context_input"] = ""
    st.session_state["mem_topics_input"] = []
    st.session_state["mem_sectors_input"] = []
    st.session_state["mem_notes_input"] = ""
    st.session_state["mem_status_notification"] = ("info", "🗑️ Memory cleared.")


def _cb_synthesise_persona(memory_store: UserMemoryStore) -> None:
    """Callback to synthesise a cohesive user persona from search interaction history."""
    persona = memory_store.synthesise_memory_profile()
    st.session_state["mem_user_context_input"] = persona
    st.session_state["mem_status_notification"] = ("success", "🤖 Synthesised legal research profile from history!")



def _cb_add_custom_tag(memory_store: UserMemoryStore) -> None:
    """Callback triggered before widget instantiation to add and activate a custom taxonomy tag."""
    tag_name = st.session_state.get("custom_tag_name_input", "").strip()
    dim_choice = st.session_state.get("custom_tag_dim_choice", "Regulatory Topic")
    if not tag_name:
        return

    if dim_choice == "Regulatory Topic":
        if tag_name not in st.session_state.setdefault("custom_topics", []):
            st.session_state["custom_topics"].append(tag_name)
        curr = list(st.session_state.get("mem_topics_input", []))
        if tag_name not in curr:
            curr.append(tag_name)
            st.session_state["mem_topics_input"] = curr
    else:
        if tag_name not in st.session_state.setdefault("custom_sectors", []):
            st.session_state["custom_sectors"].append(tag_name)
        curr = list(st.session_state.get("mem_sectors_input", []))
        if tag_name not in curr:
            curr.append(tag_name)
            st.session_state["mem_sectors_input"] = curr

    memory_store.save_memory({
        "user_context": st.session_state.get("mem_user_context_input", "").strip(),
        "preferred_topics": list(st.session_state.get("mem_topics_input", [])),
        "preferred_sectors": list(st.session_state.get("mem_sectors_input", [])),
        "notes": st.session_state.get("mem_notes_input", "").strip(),
    })
    st.session_state["custom_tag_name_input"] = ""
    st.session_state["mem_status_notification"] = ("success", f"✅ Added & activated tag: '{tag_name}'")


def _cb_import_json(memory_store: UserMemoryStore) -> None:
    """Callback triggered before widget instantiation when a JSON file is uploaded."""
    uploaded = st.session_state.get("mem_upload_file_uploader")
    if uploaded is None:
        return
    try:
        imp = json.load(uploaded)
        if isinstance(imp, dict):
            imp_context = imp.get("user_context", "")
            imp_topics = imp.get("preferred_topics", [])
            imp_sectors = imp.get("preferred_sectors", [])
            imp_notes = imp.get("notes", "")

            if not isinstance(imp_topics, list):
                imp_topics = []
            if not isinstance(imp_sectors, list):
                imp_sectors = []

            for t in imp_topics:
                if t not in st.session_state.setdefault("custom_topics", []):
                    st.session_state["custom_topics"].append(t)
            for s in imp_sectors:
                if s not in st.session_state.setdefault("custom_sectors", []):
                    st.session_state["custom_sectors"].append(s)

            saved_profile = {
                "user_context": imp_context,
                "preferred_topics": imp_topics,
                "preferred_sectors": imp_sectors,
                "notes": imp_notes,
            }
            memory_store.save_memory(saved_profile)
            st.session_state["mem_user_context_input"] = imp_context
            st.session_state["mem_topics_input"] = imp_topics
            st.session_state["mem_sectors_input"] = imp_sectors
            st.session_state["mem_notes_input"] = imp_notes
            st.session_state["mem_status_notification"] = ("success", "✅ Profile JSON successfully loaded!")
    except Exception as e:
        st.session_state["mem_status_notification"] = ("error", f"❌ Failed to parse JSON file: {e}")


def render_memory_panel(memory_store: UserMemoryStore) -> None:
    """Renders the 4-field personalised memory controls and search history in the sidebar."""
    st.sidebar.markdown("### 🧠 Personalisation Memory")

    # Display pending notification if set by any callback
    if "mem_status_notification" in st.session_state:
        kind, msg = st.session_state.pop("mem_status_notification")
        if kind == "success":
            st.sidebar.success(msg)
        elif kind == "info":
            st.sidebar.info(msg)
        elif kind == "error":
            st.sidebar.error(msg)

    # Load persisted memory from JSON
    mem_data = memory_store.load_memory()
    user_context = mem_data.get("user_context", "")
    notes = mem_data.get("notes", "")
    preferred_topics = mem_data.get("preferred_topics", [])
    preferred_sectors = mem_data.get("preferred_sectors", [])

    if not isinstance(preferred_topics, list):
        preferred_topics = []
    if not isinstance(preferred_sectors, list):
        preferred_sectors = []

    # Maintain custom tags across sessions
    if "custom_topics" not in st.session_state:
        st.session_state["custom_topics"] = []
    if "custom_sectors" not in st.session_state:
        st.session_state["custom_sectors"] = []

    # Compile available taxonomy options + persisted choices + custom tags
    tax_topics = list(config.TAXONOMY.get("Regulatory Topics", []))
    tax_sectors = list(config.TAXONOMY.get("Industry / Sector", []))

    all_topics = sorted(list(dict.fromkeys(tax_topics + preferred_topics + st.session_state["custom_topics"])))
    all_sectors = sorted(list(dict.fromkeys(tax_sectors + preferred_sectors + st.session_state["custom_sectors"])))

    # Synchronise Streamlit widget session state if not yet set
    if "mem_user_context_input" not in st.session_state:
        st.session_state["mem_user_context_input"] = user_context
    if "mem_topics_input" not in st.session_state:
        st.session_state["mem_topics_input"] = [t for t in preferred_topics if t in all_topics]
    if "mem_sectors_input" not in st.session_state:
        st.session_state["mem_sectors_input"] = [s for s in preferred_sectors if s in all_sectors]
    if "mem_notes_input" not in st.session_state:
        st.session_state["mem_notes_input"] = notes

    # Main Profile Expander
    with st.sidebar.expander("👤 User Profile & Memory Prompt", expanded=True):
        st.caption("Customise all 4 memory fields dynamically to tailor legal retrieval and LLM briefings.")

        # Field 1: User Profile & Legal Background
        st.text_area(
            "1. Legal Background / Interests",
            height=85,
            help="Your ongoing background context, organisation role, or specific area of interest",
            key="mem_user_context_input",
        )

        # Field 2: Preferred Regulatory Topics
        st.multiselect(
            "2. Priority Regulatory Topics",
            options=all_topics,
            help="Specific EU regulatory topics to prioritise during search and synthesis",
            key="mem_topics_input",
        )

        # Field 3: Preferred Industry Sectors
        st.multiselect(
            "3. Priority Industry Sectors",
            options=all_sectors,
            help="Specific business sectors or industry verticals to target",
            key="mem_sectors_input",
        )

        # Field 4: Analytical Directives
        st.text_area(
            "4. Analytical Directives",
            height=75,
            help="Custom guidelines for summary synthesis (e.g. liability caps, sanctions, enforcement timelines)",
            key="mem_notes_input",
        )

        # Action Buttons: Save & Reset using on_click callbacks
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "💾 Save Profile",
                on_click=_cb_save_profile,
                args=(memory_store,),
                use_container_width=True,
                type="primary",
                key="btn_save_mem_profile",
            )

        with col2:
            st.button(
                "🔄 Reset Defaults",
                on_click=_cb_reset_defaults,
                args=(memory_store,),
                use_container_width=True,
                key="btn_reset_mem_defaults",
            )

        # Clear All Profile Fields
        st.button(
            "🗑️ Clear All Fields",
            on_click=_cb_clear_all,
            args=(memory_store,),
            use_container_width=True,
            key="btn_clear_mem_all",
        )

        # Auto-synthesise persona from recent interaction history
        st.button(
            "🤖 Synthesise Persona from History",
            on_click=_cb_synthesise_persona,
            args=(memory_store,),
            use_container_width=True,
            help="Uses local LLM to synthesise your recent search themes into a cohesive legal persona",
            key="btn_synthesise_persona",
        )

        # Download / export memory JSON and Markdown
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.download_button(
                label="📥 JSON",
                data=memory_store.export_memory_json(),
                file_name="prompt_context.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_exp2:
            st.download_button(
                label="📄 Markdown",
                data=memory_store.export_memory_markdown(),
                file_name="prompt_context.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Collapsible Custom Tag Additions
    with st.sidebar.expander("➕ Add Custom Topic or Sector", expanded=False):
        st.text_input(
            "Custom Tag Name",
            placeholder="e.g. Automotive, Space Law",
            key="custom_tag_name_input",
        )
        st.radio(
            "Taxonomy Dimension",
            ["Regulatory Topic", "Industry Sector"],
            horizontal=True,
            key="custom_tag_dim_choice",
        )
        st.button(
            "Add & Activate Tag",
            on_click=_cb_add_custom_tag,
            args=(memory_store,),
            use_container_width=True,
            key="btn_add_custom_tag",
        )

    # Collapsible JSON Import
    with st.sidebar.expander("📤 Import Profile JSON", expanded=False):
        st.file_uploader(
            "Upload prompt_context.json",
            type=["json"],
            key="mem_upload_file_uploader",
            on_change=_cb_import_json,
            args=(memory_store,),
        )

    # Active Memory Summary Badges
    with st.sidebar.expander("🔍 Active Memory Preview", expanded=False):
        st.markdown(f"**Last Updated:** `{mem_data.get('updated_at', 'N/A')[:19].replace('T', ' ')}`")

        badges_html = []
        for t in st.session_state.get("mem_topics_input", preferred_topics):
            badges_html.append(f'<span class="badge badge-topic">⚖️ {t}</span>')
        for s in st.session_state.get("mem_sectors_input", preferred_sectors):
            badges_html.append(f'<span class="badge badge-sector">🏢 {s}</span>')

        if badges_html:
            st.markdown(f"<div>{''.join(badges_html)}</div>", unsafe_allow_html=True)
        else:
            st.caption("No topics or sectors selected.")

        curr_ctx = st.session_state.get("mem_user_context_input", "")
        if curr_ctx:
            preview_ctx = curr_ctx[:120] + "..." if len(curr_ctx) > 120 else curr_ctx
            st.markdown(f"**Profile:** *{preview_ctx}*")

        curr_notes = st.session_state.get("mem_notes_input", "")
        if curr_notes:
            preview_notes = curr_notes[:120] + "..." if len(curr_notes) > 120 else curr_notes
            st.markdown(f"**Directives:** *{preview_notes}*")

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
