"""Interactive Plotly chart visualizations for 3-axis radar chart and tag histograms."""

from typing import Any, Dict, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import SearchAnalytics


def render_analytics_dashboard(results: List[Dict[str, Any]]) -> None:
    """Renders the side-by-side radar chart and tag distribution histogram."""
    if not results:
        return

    analytics = SearchAnalytics()
    hist_df = analytics.prepare_histogram_df(results)
    radar_data = analytics.prepare_radar_data(results)

    st.markdown("### 📊 Legal Multi-Dimensional Analytics")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Radar Chart
        st.markdown("##### 🧭 3-Axis Domain Radar")
        fig_radar = go.Figure()

        dim_colors = {
            "Jurisdiction": "rgba(139, 92, 246, 0.75)",      # Violet
            "Regulatory Topics": "rgba(245, 158, 11, 0.75)",  # Amber
            "Industry / Sector": "rgba(6, 182, 212, 0.75)",   # Cyan
        }
        dim_fills = {
            "Jurisdiction": "rgba(139, 92, 246, 0.25)",
            "Regulatory Topics": "rgba(245, 158, 11, 0.25)",
            "Industry / Sector": "rgba(6, 182, 212, 0.25)",
        }

        has_data = False
        for dim, data in radar_data.items():
            cats = data["categories"]
            vals = data["values"]
            if not cats or max(vals, default=0) == 0:
                continue

            has_data = True
            # Close the polygon loop for polar plot
            r_vals = vals + [vals[0]]
            theta_vals = cats + [cats[0]]

            fig_radar.add_trace(go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                name=dim,
                line=dict(color=dim_colors.get(dim, "#6366f1"), width=2),
                fillcolor=dim_fills.get(dim, "rgba(99, 102, 241, 0.2)"),
            ))

        if has_data:
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(15, 23, 42, 0.8)",
                    radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(color="#94a3b8", size=9), gridcolor="rgba(255, 255, 255, 0.1)"),
                    angularaxis=dict(tickfont=dict(color="#cbd5e1", size=10), gridcolor="rgba(255, 255, 255, 0.1)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#f8fafc"),
                margin=dict(l=40, r=40, t=30, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=10)),
                height=380,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Insufficient taxonomy tags across results to generate radar projection.")

    with col2:
        # Tag Distribution Histogram
        st.markdown("##### 📈 Category Distribution Histogram")
        if not hist_df.empty:
            color_map = {
                "Jurisdiction": "#8b5cf6",
                "Regulatory Topics": "#f59e0b",
                "Industry / Sector": "#06b6d4",
            }
            fig_bar = px.bar(
                hist_df,
                x="Category",
                y="Count",
                color="Dimension",
                color_discrete_map=color_map,
                text="Count",
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15, 23, 42, 0.5)",
                font=dict(family="Plus Jakarta Sans, sans-serif", color="#f8fafc"),
                margin=dict(l=20, r=20, t=30, b=30),
                xaxis=dict(tickangle=-35, tickfont=dict(size=10), gridcolor="rgba(255, 255, 255, 0.05)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5, font=dict(size=10)),
                height=380,
            )
            fig_bar.update_traces(textposition="outside", marker_line_width=1, marker_line_color="rgba(255,255,255,0.2)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No distribution data available.")
