"""
dashboard/app.py — TrazIT Investigation Dashboard

SIH-ready: evidence-first, projector-readable, real data only.
Light theme, IBM Plex Sans typography, risk-severity accents.

Run: streamlit run app.py  (from inside dashboard/)
"""
from __future__ import annotations

import sys
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "pipeline"))
sys.path.insert(0, os.path.join(_HERE, "..", "llm"))
from inference import InferenceContext, run_inference, get_upload_entity_evidence

st.set_page_config(
    page_title="TrazIT — Bitcoin Transaction Forensics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Risk severity palette — reserved for risk levels, never decoration
# ---------------------------------------------------------------------------
RISK_COLORS = {"CRITICAL": "#B03A2E", "HIGH": "#D35400", "MEDIUM": "#D4AC0D", "LOW": "#95A5A6"}
RISK_BG     = {"CRITICAL": "#FADBD8", "HIGH": "#FDEBD0", "MEDIUM": "#FEF9E7", "LOW": "#F2F3F4"}

# ---------------------------------------------------------------------------
# Custom CSS — minimal, targets stable selectors
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}
code, .stCode, pre {
    font-family: 'IBM Plex Mono', 'Cascadia Mono', 'Consolas', monospace;
}
[data-testid="stHeader"] { display: none !important; }
header { display: none !important; visibility: hidden !important; height: 0 !important; }
h1 { font-weight: 700; font-size: 1.6rem; color: #1A1A2E; letter-spacing: -0.01em; }
h2, h3 { font-weight: 600; color: #2D2D3F; }
[data-testid="stMetricValue"] {
    font-size: 1.8rem; font-weight: 700;
    font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem; font-weight: 500; text-transform: uppercase;
    letter-spacing: 0.04em; color: #6C6C80;
}
.block-container { padding-top: 0.5rem; max-width: 1200px; }
.risk-badge {
    display: inline-block; padding: 3px 12px; border-radius: 3px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em;
    color: white; vertical-align: middle;
}
.risk-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 4px; font-weight: 600;
    font-size: 0.85rem;
}
.entity-addr {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; color: #2D2D3F;
}
.section-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #8C8C9E; margin-bottom: 4px;
}
.evidence-item {
    padding: 8px 12px; margin: 4px 0; border-left: 3px solid #D5D5E0;
    background: #F8F7F4; font-size: 0.88rem; line-height: 1.5;
}
hr { border: none; border-top: 1px solid #E0DED8; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, "#95A5A6")
    return f'<span class="risk-badge" style="background:{color}">{level}</span>'

def risk_chip(level: str, count: int) -> str:
    color = RISK_COLORS.get(level, "#95A5A6")
    bg = RISK_BG.get(level, "#F2F3F4")
    return (f'<span class="risk-chip" style="background:{bg};color:{color}">'
            f'<strong>{count}</strong> {level}</span>')

def try_load_llm():
    try:
        from query_llm import EvidenceLLM, find_model_file
    except ImportError:
        return None, "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
    if find_model_file() is None:
        return None, "No .gguf model file found in llm/models/. See llm/README.md."
    try:
        return EvidenceLLM(), None
    except Exception as e:
        return None, f"Model failed to load: {e}"


@st.cache_resource(show_spinner=False)
def get_inference_context():
    return InferenceContext()


# ---------------------------------------------------------------------------
# Boot — load model artifacts
# ---------------------------------------------------------------------------
try:
    ctx = get_inference_context()
except FileNotFoundError as e:
    logging.error("Model artifacts not found: %s", e)
    st.error("Required model files are missing. Verify that ml/artifacts/ contains the trained model.")
    st.stop()
except Exception as e:
    logging.error("Failed to load model context: %s", e, exc_info=True)
    st.error("The scoring model could not be loaded. Check that model artifacts are valid.")
    st.stop()

# ---------------------------------------------------------------------------
# Header + theme toggle
# ---------------------------------------------------------------------------
hdr_left, hdr_right = st.columns([6, 1])
with hdr_left:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px;">
        <svg width="32" height="32" viewBox="0 0 32 32" xmlns="http://www.w3.org/2000/svg">
            <circle cx="16" cy="16" r="16" fill="#F7931A"/>
            <path d="M22.4 14.1c.3-2-1.2-3.1-3.3-3.8l.7-2.7-1.6-.4-.7 2.6c-.4-.1-.8-.2-1.3-.3l.7-2.7-1.6-.4-.7 2.7c-.3-.1-.7-.2-1-.3l-2.3-.6-.4 1.7s1.2.3 1.2.3c.7.2.8.6.8 1l-.8 3.2c0 0 .1 0 .1 0l-.1 0-1.1 4.5c-.1.2-.3.5-.7.4 0 0-1.2-.3-1.2-.3l-.8 1.8 2.1.5c.4.1.8.2 1.2.3l-.7 2.8 1.6.4.7-2.7c.4.1.9.2 1.3.3l-.7 2.7 1.6.4.7-2.8c3 .6 5.2.3 6.2-2.4.7-2.2 0-3.4-1.6-4.2 1.1-.3 2-.1 2.2-1.9zm-4 5.5c-.5 2.1-4.2 1-5.4.7l1-3.9c1.2.3 5 .9 4.4 3.2zm.5-5.6c-.5 1.9-3.5.9-4.5.7l.9-3.5c1 .3 4.2.7 3.6 2.8z" fill="white"/>
        </svg>
        <h1 style="margin:0; padding:0; font-weight:700; font-size:1.6rem; color:#1A1A2E;">TrazIT</h1>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Bitcoin Transaction Forensics")
with hdr_right:
    st.markdown("")  # vertical spacing
    dark_mode = st.toggle("Dark mode", value=False, key="dark_mode")

# Apply dark mode overrides when toggled
if dark_mode:
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stApp"],
    .main, .block-container, [data-testid="stHeader"] {
        background-color: #0E1117 !important; color: #FAFAFA !important;
    }
    [data-testid="stSidebar"] { background-color: #161B22 !important; }
    h1, h2, h3, p, span, label, .stMarkdown, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], .section-label, .entity-addr {
        color: #FAFAFA !important;
    }
    [data-testid="stMetricLabel"] { color: #8C8C9E !important; }
    .evidence-item { background: #1A1D24 !important; border-left-color: #333 !important; color: #E0E0E0 !important; }
    .risk-chip { background: rgba(255,255,255,0.08) !important; }
    hr { border-top-color: #2A2D35 !important; }
    [data-testid="stDataFrame"] { color-scheme: dark; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
st.markdown('<p class="section-label">Upload transaction data</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose file", type=["csv", "json", "xml"], label_visibility="collapsed"
)

if uploaded_file is not None:
    tmp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
    try:
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except OSError as e:
        logging.error("Failed to save uploaded file: %s", e)
        st.error("Could not save file to disk. Check available disk space.")
        st.stop()

    if st.session_state.get("_last_uploaded_name") != uploaded_file.name:
        try:
            with st.spinner("Scoring upload against the trained model..."):
                st.session_state["scored"] = run_inference(tmp_path, ctx)
        except ValueError as e:
            logging.warning("Data validation error: %s", e)
            st.error(f"File could not be processed: {e}")
            st.session_state.pop("scored", None)
            st.session_state.pop("_last_uploaded_name", None)
            st.stop()
        except (KeyError, TypeError, IndexError) as e:
            logging.error("Data processing error: %s", e, exc_info=True)
            st.error("Data structure does not match the expected format.")
            st.session_state.pop("scored", None)
            st.session_state.pop("_last_uploaded_name", None)
            st.stop()
        except Exception as e:
            logging.error("Unexpected pipeline error: %s", e, exc_info=True)
            st.error("An unexpected error occurred. Please try a different file.")
            st.session_state.pop("scored", None)
            st.session_state.pop("_last_uploaded_name", None)
            st.stop()
        st.session_state["_last_uploaded_name"] = uploaded_file.name
        for k in ("risk_level_filter", "min_score_filter", "search_filter", "country_filter"):
            st.session_state.pop(k, None)

result = st.session_state.get("scored")

if result is None:
    st.info("Upload a file above to begin analysis.")
    st.stop()

if not result["success"]:
    st.error(f"Upload failed validation: {result['error']}")
    st.code(result["validation_report"])
    st.stop()

# ---------------------------------------------------------------------------
# Post-upload: extract data
# ---------------------------------------------------------------------------
risk_df = result["risk_df"]
if "risk_score" not in risk_df.columns or "risk_level" not in risk_df.columns:
    st.error(f"Version mismatch: missing columns. Found: {list(risk_df.columns)}")
    st.stop()

st.markdown("---")

# Ingestion summary — compact, not a separate section
col_s1, col_s2 = st.columns([3, 1])
with col_s1:
    st.success(
        f"**{result['n_rows_valid']}** valid transactions from "
        f"**{result['n_entities']}** entities scored successfully."
    )
with col_s2:
    with st.expander("Validation details"):
        st.code(result["validation_report"])

# Drift notice
drift = result.get("drift_result")
if drift and drift["drift_detected"]:
    if drift.get("severity") == "severe":
        st.warning("Distribution drift detected. Scores may have reduced confidence.")
    else:
        st.info("Minor distributional variation. Scores remain reliable.")

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_overview, tab_investigation, tab_clusters = st.tabs([
    "Overview", "Entity Investigation", "Clusters & Communities"
])

# ============================= TAB 1: OVERVIEW ============================
with tab_overview:

    # Risk distribution chips
    level_counts = risk_df["risk_level"].value_counts()
    chips_html = " &nbsp; ".join(
        risk_chip(lv, int(level_counts.get(lv, 0)))
        for lv in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    )
    st.markdown(chips_html, unsafe_allow_html=True)

    st.markdown("")  # spacing

    # Key metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions", result["n_rows_valid"])
    m2.metric("Entities", result["n_entities"])
    m3.metric("Clusters", result["n_clusters_found"])
    m4.metric("Communities", result["n_communities_found"])

    st.markdown("---")

    # Top flagged entities
    st.markdown("### Flagged for Investigation")

    flagged = risk_df[risk_df["risk_level"].isin(["CRITICAL", "HIGH"])].sort_values(
        "risk_score", ascending=False
    )
    if len(flagged) > 0:
        flagged_display = flagged[[
            "entity_id", "risk_score", "risk_level", "primary_country",
            "anomaly_component", "graph_centrality_component",
            "cluster_outlier_component", "flagged_neighbor_component",
        ]].copy()
        flagged_display.columns = [
            "Entity", "Score", "Level", "Country",
            "Anomaly", "Centrality", "Outlier", "Flagged Nbrs",
        ]
        flagged_display["Score"] = flagged_display["Score"].round(1)
        for c in ["Anomaly", "Centrality", "Outlier", "Flagged Nbrs"]:
            flagged_display[c] = flagged_display[c].round(3)
        st.dataframe(flagged_display, height=min(400, 35 * len(flagged_display) + 38),
                      hide_index=True, use_container_width=True)
    else:
        st.info("No entities flagged as CRITICAL or HIGH in this dataset.")

    st.markdown("---")

    # Full entity table with filters
    st.markdown("### All Entities")

    fc1, fc2, fc3, fc4 = st.columns([2, 2, 2, 2])
    with fc1:
        levels = st.multiselect(
            "Risk level", ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"], key="risk_level_filter",
        )
    with fc2:
        min_score = st.slider("Min score", 0, 100, 0, key="min_score_filter")
    with fc3:
        available_countries = sorted({c for cs in risk_df["countries"] for c in cs})
        selected_countries = st.multiselect("Country", available_countries, default=[], key="country_filter")
    with fc4:
        search = st.text_input("Search entity", key="search_filter")

    filtered = risk_df[risk_df["risk_level"].isin(levels)]
    filtered = filtered[filtered["risk_score"] >= min_score]
    if selected_countries:
        filtered = filtered[filtered["countries"].apply(
            lambda cs: any(c in selected_countries for c in cs)
        )]
    if search:
        filtered = filtered[filtered["entity_id"].str.contains(search, case=False, na=False)]
    if len(filtered) > 0:
        filtered = filtered.sort_values("risk_score", ascending=False)

    st.caption(f"**{len(filtered)}** entities match filters")

    if len(filtered) == 0:
        st.info("No entities match the current filters. Adjust risk level, score, country, or search.")
    else:
        display_df = filtered[[
            "entity_id", "risk_score", "risk_level", "primary_country",
            "anomaly_component", "graph_centrality_component",
            "cluster_outlier_component", "flagged_neighbor_component",
        ]].copy()
        display_df.columns = [
            "Entity", "Score", "Level", "Country",
            "Anomaly", "Centrality", "Outlier", "Flagged Nbrs",
        ]
        display_df["Score"] = display_df["Score"].round(1)
        for c in ["Anomaly", "Centrality", "Outlier", "Flagged Nbrs"]:
            display_df[c] = display_df[c].round(3)
        st.dataframe(display_df, height=450, hide_index=True, use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Export filtered results (CSV)", csv, "filtered_alerts.csv", "text/csv")


# ========================= TAB 2: INVESTIGATION ==========================
with tab_investigation:
    entity_id = st.selectbox(
        "Select entity",
        risk_df.sort_values("risk_score", ascending=False)["entity_id"].tolist(),
        help="Sorted by risk score, highest first.",
    )
    row = risk_df[risk_df["entity_id"] == entity_id].iloc[0]

    st.markdown("---")

    # ---- Wallet Profile ----
    st.markdown('<p class="section-label">Wallet Profile</p>', unsafe_allow_html=True)

    cluster_val = int(row["behavioral_cluster"]) if row["behavioral_cluster"] != -1 else "—"
    community_val = int(row["graph_community"]) if not (row["graph_community"] != row["graph_community"]) else "—"

    st.markdown(f"""
    <div style="padding:16px 20px; background:{'#1A1D24' if dark_mode else '#F0EFEB'}; border-radius:6px; margin-bottom:12px;">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <div>
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;
                            color:#8C8C9E; margin-bottom:4px;">Wallet Address</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.95rem; word-break:break-all;
                            color:{'#FAFAFA' if dark_mode else '#1A1A2E'};">{entity_id}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;
                            color:#8C8C9E; margin-bottom:4px;">Risk Score</div>
                <div style="font-size:1.6rem; font-weight:700; color:{'#FAFAFA' if dark_mode else '#1A1A2E'};">
                    {row['risk_score']:.1f}<span style="font-size:0.8rem; color:#8C8C9E;"> / 100</span>
                </div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:16px; margin-top:12px; flex-wrap:wrap;">
            {risk_badge(row['risk_level'])}
            <span style="font-size:0.82rem; color:{'#CCCCCC' if dark_mode else '#555555'};">
                Country: <strong>{row['primary_country']}</strong>
            </span>
            <span style="font-size:0.82rem; color:{'#CCCCCC' if dark_mode else '#555555'};">
                Cluster: <strong>{cluster_val}</strong>
            </span>
            <span style="font-size:0.82rem; color:{'#CCCCCC' if dark_mode else '#555555'};">
                Community: <strong>{community_val}</strong>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ---- Component breakdown ----
    comp_col, graph_col = st.columns([2, 3])

    with comp_col:
        st.markdown('<p class="section-label">Risk Score Components</p>', unsafe_allow_html=True)
        components = {
            "Anomaly":          row["anomaly_component"],
            "Graph Centrality": row["graph_centrality_component"],
            "Cluster Outlier":  row["cluster_outlier_component"],
            "Flagged Neighbors": row["flagged_neighbor_component"],
        }
        comp_colors = ["#2E86C1", "#27AE60", "#8E44AD", "#D35400"]
        comp_fig = go.Figure(go.Bar(
            x=list(components.keys()),
            y=list(components.values()),
            marker_color=comp_colors,
            text=[f"{v:.3f}" for v in components.values()],
            textposition="outside",
        ))
        comp_fig.update_layout(
            yaxis_range=[0, 1], height=280,
            margin=dict(t=10, b=10, l=10, r=10),
            template="plotly_dark" if dark_mode else "plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)" if dark_mode else "rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#333" if dark_mode else "#E8E8E8"),
            font=dict(color="#FAFAFA" if dark_mode else "#1A1A2E"),
        )
        st.plotly_chart(comp_fig, use_container_width=True)

    # ---- Graph neighborhood ----
    with graph_col:
        st.markdown('<p class="section-label">Graph Neighborhood</p>', unsafe_allow_html=True)
        H = result["H"]
        G = result.get("G")
        if entity_id in H and len(list(H.neighbors(entity_id))) > 0:
            all_neighbors = list(H.neighbors(entity_id))

            neighbor_weight = {}
            if G is not None and entity_id in G:
                for nbr in all_neighbors:
                    w = 0
                    for u, v, k in G.edges(keys=True):
                        if k != "transfer":
                            continue
                        if (u == entity_id and v == nbr) or (u == nbr and v == entity_id):
                            w += 1
                    neighbor_weight[nbr] = w
            else:
                for nbr in all_neighbors:
                    neighbor_weight[nbr] = 1

            ranked = sorted(all_neighbors, key=lambda n: neighbor_weight.get(n, 0), reverse=True)
            MAX_NEIGHBORS = 10
            neighbors = ranked[:MAX_NEIGHBORS]
            sub_nodes = [entity_id] + neighbors

            predecessors, successors = set(), set()
            if G is not None and entity_id in G:
                for u, v, k in G.in_edges(entity_id, keys=True):
                    if k == "transfer" and u in sub_nodes:
                        predecessors.add(u)
                for u, v, k in G.out_edges(entity_id, keys=True):
                    if k == "transfer" and v in sub_nodes:
                        successors.add(v)

            import random
            rng = random.Random(42)
            pos = {entity_id: (0.0, 0.0)}
            both = predecessors & successors
            only_pred = predecessors - both
            only_succ = successors - both
            undirected_only = set(neighbors) - predecessors - successors

            max_side_count = max(len(only_pred), len(only_succ), len(both), len(undirected_only), 1)
            y_step = min(0.35, 2.5 / max_side_count)

            def _spread_y(nodes, x_val, y_offset=0.0):
                sorted_nodes = sorted(nodes)
                n = len(sorted_nodes)
                for i, nd in enumerate(sorted_nodes):
                    y = ((i - (n - 1) / 2) * y_step + y_offset) if n > 0 else y_offset
                    pos[nd] = (x_val + rng.uniform(-0.04, 0.04), y + rng.uniform(-0.04, 0.04))

            _spread_y(only_pred, -1.0)
            _spread_y(only_succ, 1.0)
            _spread_y(both, 0.0, y_offset=1.0)
            _spread_y(undirected_only, 0.0, y_offset=-1.0)

            edge_x, edge_y = [], []
            for nbr in neighbors:
                if nbr not in pos:
                    continue
                x0, y0 = pos[entity_id]
                x1, y1 = pos[nbr]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1.2, color="#C0C0C0"), hoverinfo="none",
            )

            node_ids = sub_nodes
            node_levels = [
                row["risk_level"] if n == entity_id else
                risk_df.loc[risk_df["entity_id"] == n, "risk_level"].values[0]
                if n in risk_df["entity_id"].values else "LOW"
                for n in node_ids
            ]

            def _role(n):
                if n == entity_id:
                    return "SELECTED"
                if n in both:
                    return "sends & receives"
                if n in predecessors:
                    return "sender"
                if n in successors:
                    return "receiver"
                return "connected"

            nbr_ids = [n for n in node_ids if n != entity_id]
            nbr_levels = [node_levels[node_ids.index(n)] for n in nbr_ids]
            nbr_x = [pos[n][0] for n in nbr_ids]
            nbr_y = [pos[n][1] for n in nbr_ids]
            nbr_colors = [RISK_COLORS.get(lv, "#95A5A6") for lv in nbr_levels]
            nbr_sizes = [max(8, min(18, 8 + neighbor_weight.get(n, 0) * 2)) for n in nbr_ids]

            neighbor_trace = go.Scatter(
                x=nbr_x, y=nbr_y, mode="markers", hoverinfo="text",
                text=[f"{n[:16]}...<br>{lv} ({_role(n)})<br>{neighbor_weight.get(n,0)} transfers"
                      for n, lv in zip(nbr_ids, nbr_levels)],
                marker=dict(size=nbr_sizes, color=nbr_colors, symbol="circle",
                            line=dict(width=1.5, color="#CCCCCC" if dark_mode else "#444444")),
            )

            sel_x, sel_y = pos[entity_id]
            sel_color = RISK_COLORS.get(row["risk_level"], "#95A5A6")

            halo_trace = go.Scatter(
                x=[sel_x], y=[sel_y], mode="markers", hoverinfo="skip",
                marker=dict(size=42, color="rgba(255,255,255,0.06)" if dark_mode else "rgba(0,0,0,0.04)",
                            symbol="diamond",
                            line=dict(width=2, color="rgba(255,255,255,0.15)" if dark_mode else "rgba(0,0,0,0.12)")),
            )
            selected_trace = go.Scatter(
                x=[sel_x], y=[sel_y], mode="markers", hoverinfo="text",
                text=[f"{entity_id[:20]}...<br>{row['risk_level']} (SELECTED)<br>{len(neighbors)} neighbors"],
                marker=dict(size=28, color=sel_color, symbol="diamond",
                            line=dict(width=3, color="#FFFFFF" if dark_mode else "#1A1A2E")),
            )

            net_fig = go.Figure([edge_trace, neighbor_trace, halo_trace, selected_trace])

            net_fig.add_annotation(
                x=sel_x, y=sel_y + 0.18, text="SELECTED", showarrow=False,
                font=dict(color="#FFFFFF" if dark_mode else "#1A1A2E", size=10,
                          family="IBM Plex Sans, sans-serif"),
                bgcolor="rgba(0,0,0,0.6)" if dark_mode else "rgba(255,255,255,0.8)",
                borderpad=3, bordercolor="#555" if dark_mode else "#C0C0C0",
            )

            if G is not None:
                for nbr in neighbors:
                    if nbr not in pos:
                        continue
                    for u, v, k in G.out_edges(entity_id, keys=True):
                        if k == "transfer" and v == nbr:
                            x0, y0 = pos[entity_id]
                            x1, y1 = pos[nbr]
                            net_fig.add_annotation(
                                x=x1, y=y1, ax=x0, ay=y0,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=3, arrowsize=1.2,
                                arrowwidth=1, arrowcolor="#AAAAAA", opacity=0.6,
                            )
                            break
                    for u, v, k in G.in_edges(entity_id, keys=True):
                        if k == "transfer" and u == nbr:
                            x0, y0 = pos[nbr]
                            x1, y1 = pos[entity_id]
                            net_fig.add_annotation(
                                x=x1, y=y1, ax=x0, ay=y0,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=3, arrowsize=1.2,
                                arrowwidth=1, arrowcolor="#AAAAAA", opacity=0.6,
                            )
                            break

            net_fig.update_layout(
                showlegend=False, height=350,
                margin=dict(t=15, b=15, l=10, r=10),
                xaxis=dict(visible=False, range=[-1.5, 1.5]),
                yaxis=dict(visible=False),
                template="plotly_dark" if dark_mode else "plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1A1D24" if dark_mode else "#F5F4F0",
            )
            st.plotly_chart(net_fig, use_container_width=True)
        else:
            st.info("No direct graph neighbors in the uploaded data.")

    # ---- Evidence ----
    st.markdown("---")
    st.markdown("### Why was this entity flagged?")
    try:
        evidence = get_upload_entity_evidence(entity_id, result)
        if evidence["n_evidence_items"] == 0:
            st.info(evidence.get("note", "No significant evidence signals found."))
        else:
            category_labels = {
                "feature": "Behavioral Feature Deviations",
                "model_output": "Anomaly Model Output",
                "graph": "Graph Evidence",
                "behavioral": "Behavioral Clustering",
            }
            by_category = {}
            for item in evidence["evidence"]:
                by_category.setdefault(item["category"], []).append(item["statement"])

            for cat, statements in by_category.items():
                st.markdown(f"**{category_labels.get(cat, cat)}**")
                for s in statements:
                    st.markdown(
                        f'<div class="evidence-item">{s}</div>',
                        unsafe_allow_html=True,
                    )
    except Exception as e:
        logging.error("Evidence generation failed for %s: %s", entity_id, e, exc_info=True)
        st.warning("Evidence details could not be generated for this entity.")

    # ---- LLM Q&A ----
    st.markdown("---")
    st.markdown("""
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:8px;">
        <div style="width:44px; height:44px; border-radius:50%; 
                    background:linear-gradient(135deg,#2E86C1,#1A5276); 
                    display:flex; align-items:center; justify-content:center;
                    font-weight:700; color:white; font-size:0.85rem;
                    letter-spacing:0.03em; flex-shrink:0;">AI</div>
        <div>
            <div style="font-weight:600; font-size:1.05rem; color:#1A1A2E;">TrazIT Assistant</div>
            <div style="font-size:0.82rem; color:#6C6C80;">How can I help you investigate this entity?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    llm, llm_error = try_load_llm()
    if llm_error:
        st.info(f"LLM not available: {llm_error}")
    else:
        question = st.text_input("Ask a question", key=f"llm_q_{entity_id}",
                                  placeholder="e.g. What makes this entity suspicious?",
                                  label_visibility="collapsed")
        if question and st.button("Ask", type="primary"):
            try:
                with st.spinner("Thinking..."):
                    answer = llm.answer_question(question, evidence)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                logging.error("LLM query failed: %s", e, exc_info=True)
                st.warning("Could not answer that question. Try rephrasing.")


# ====================== TAB 3: CLUSTERS & COMMUNITIES ====================
with tab_clusters:
    st.caption(
        f"{result['n_clusters_found']} behavioral clusters, "
        f"{result['n_communities_found']} graph communities identified."
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Behavioral Clusters")
        cs = risk_df.groupby("behavioral_cluster").agg(
            n=("entity_id", "size"),
            avg_risk=("risk_score", "mean"),
            flagged=("risk_level", lambda s: s.isin(["CRITICAL", "HIGH"]).sum()),
        ).reset_index()
        cs["behavioral_cluster"] = cs["behavioral_cluster"].apply(
            lambda x: "Unclustered" if x == -1 else f"Group {int(x)}"
        )
        cs["avg_risk"] = cs["avg_risk"].round(1)
        cs.columns = ["Group", "Entities", "Avg Risk", "Flagged"]
        st.dataframe(cs.sort_values("Entities", ascending=False),
                      hide_index=True, use_container_width=True)

    with c2:
        st.markdown("### Graph Communities")
        comm = risk_df.dropna(subset=["graph_community"]).groupby("graph_community").agg(
            n=("entity_id", "size"),
            avg_risk=("risk_score", "mean"),
            flagged=("risk_level", lambda s: s.isin(["CRITICAL", "HIGH"]).sum()),
        ).reset_index()
        if len(comm):
            comm["graph_community"] = comm["graph_community"].apply(lambda x: f"Community {int(x)}")
            comm["avg_risk"] = comm["avg_risk"].round(1)
            comm.columns = ["Community", "Entities", "Avg Risk", "Flagged"]
            st.dataframe(comm.sort_values("Entities", ascending=False),
                          hide_index=True, use_container_width=True)
        else:
            st.info("No communities detected (graph too small or sparse).")
