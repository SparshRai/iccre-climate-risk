MODEL_VERSION    = "DEMO-1.1"
PARAMETER_VERSION = "ClimateParams_v1.4_final_logic_corrected"
NGFS_DATA_VERSION = "NGFS_PhaseIII_2023"
ENGINE_BUILD      = "IntegratedClimateCreditEngine"
MODEL_BUILD_DATE  = "2026-05-06"
MODEL_HASH        = "ICCRE_demo_public_v1_1"

# ============================================================
# PRODUCT & BRANDING
# ============================================================
PRODUCT_TAGLINE  = "India's first quantitative climate-to-credit engine"
PRODUCT_SUBTITLE = "NGFS-aligned · BRSR-native · ISSB S2-ready · INR-denominated"
CONTACT_EMAIL    = "hello@iccre.in"
PRODUCT_URL      = "https://iccre.in"
LINKEDIN_URL     = "https://linkedin.com/company/iccre"
MODEL_USE_NOTE  = "Decision-support / scenario-analysis tool. Outputs require user review and calibration before regulated credit decisions."
MODEL_CONFIDENCE_DEFAULT = "Medium — assumption-led, scenario-consistent, not borrower-calibrated"
# Public demo defaults for more conservative, market-ready interpretation.
# Physical EAD allocation defaults to company-wide proportional exposure when asset register covers
# only part of company revenue; full-EAD allocation remains available as conservative advanced mode.
DEFAULT_PHYSICAL_EAD_MODE = "Company-wide proportional exposure"

# ── Demo dataset: Bharat Steel Industries Ltd (fictional) ──
# Pre-configured to produce visually compelling, realistic demo output.
# PD and DSCR stress are shown under the selected public scenarios.
# Carbon burden 11.4% of revenue. ECL ₹394 Cr vs ₹81 Cr baseline.
DEMO_DATASET = {
    "company_name":      "Bharat Steel Industries Ltd",
    "sector":            "Steel",
    "reporting_year":    2035,
    "revenue_0":         18000.0,
    "ebitda_margin_0":   0.22,
    "interest_payment":  1100.0,
    "total_assets":      45000.0,
    "exposure_at_default": 12000.0,
    "scope1":            4200000.0,
    "scope2":            1800000.0,
    "scope3":            2000000.0,
    "high_carbon_assets": 8000.0,
    "base_pd":           0.015,
    "lgd_0":             0.45,
    "planned_capex":     1800.0,
    "abatement_cost":    4500.0,
    "abatement_potential": 0.30,
    "carbon_pass_through": 0.35,
    "demand_elasticity": -0.40,
}

# ============================================================
# CHANGELOG v1.1 → v1.2
# ============================================================
# DEMO-1.1  Public demo patch: sidebar text/button contrast fixed;
#           Streamlit Plotly duplicate element IDs prevented with
#           automatic unique chart keys; peak/worst metric cards now
#           show the scenario year and scenario name; reporting-year
#           selector retained for snapshot outputs.
#
# UI-01  Complete dark-teal design system (CSS variables,
#        gradient backgrounds, card components, KPI panels).
#        All tables, charts, and metrics use consistent theme.
#
# UI-02  Professional Plotly chart styling engine applied
#        globally — dark backgrounds, teal/amber/coral palette,
#        hover tooltips, reference lines, annotations.
#
# UI-03  Dashboard landing tab: 6 KPI hero cards, risk gauge,
#        PD sparkline, module status badges.
#
# UI-04  All st.dataframe() calls replaced with styled
#        tables using background-gradient, color-coded cells,
#        and proper column formatting.
#
# BRSR-01  BRSR tab rebuilt as financial risk engine:
#          GHG intensity benchmarked against SEBI sector P25/P50/P75.
#          BRSR flags mapped to PD basis-point overlay (additive, ≤150bps).
#          3-year emissions intensity trend scoring.
#          Water stranded cost projection (5-yr, stress escalation).
#          Energy transition risk (fossil carbon surcharge 2030).
#          SEBI BRSR Core regulatory readiness score (8-point).
#          5 professional Plotly charts (benchmark bars, PD tornado,
#          readiness radar, 5-yr cost stack, emissions trend line).
#          Live integration block linking BRSR risk overlay to transition PD.
#
# PHYS-01  Physical risk methodology corrected:
#          Illustrative non-linear physical damage multiplier: 1+0.20·ΔT+0.04·ΔT²
#          (replaces flat 1+0.25·ΔT multiplier with no basis).
#          ΔT extracted directly from NGFS scenario temperature pathways
#          (same df_long as transition engine — fully aligned).
#          P10/P50/P90 uncertainty bands: σ=0.08·ΔT.
#          Signed DSCR gap consistent with FIX-04.
#          Chronic vs acute physical risk split.
#          4-panel scenario dashboard per NGFS scenario.
#          Asset vulnerability heatmap (flood/heat/cyclone).
#
# INT-01   Integrated risk tab rebuilt with rigorous methodology:
#          Gaussian copula joint PD shown with full derivation steps.
#          Risk decomposition: transition % vs physical % of total ECL.
#          BRSR risk overlay added as third additive layer on top.
#          Scenario stress table: all 3 NGFS × 6 key metrics.
#          Capital adequacy section with ICAAP capital buffer calc.
#          Board-level RAG summary (Red/Amber/Green with thresholds).
#          All v1.1 fixes (FIX-01 to FIX-07) preserved in full.
#
# All v1.1 fixes (FIX-01 to FIX-07) are fully preserved.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, json
import urllib.request
from groq import Groq
from math import radians, sin, cos, asin, sqrt

# Rasterio is required only for the Physical Risk flood layer.
# Keeping this import guarded prevents the full app from failing during deployment
# if native geospatial wheels are temporarily unavailable.
try:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.windows import Window
    RASTERIO_IMPORT_ERROR = None
except Exception as _rasterio_error:
    rasterio = None
    rowcol = None
    Window = None
    RASTERIO_IMPORT_ERROR = str(_rasterio_error)

import folium
from streamlit_folium import st_folium
from scipy.stats import norm, multivariate_normal
from datetime import datetime
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# STREAMLIT CHART KEY GUARD
# ============================================================
# Streamlit 1.5x can raise StreamlitDuplicateElementId when identical
# Plotly figures are rendered in cached views, repeated tabs, or reruns
# without explicit keys. This wrapper adds a unique deterministic key
# when the caller has not provided one, while preserving all explicit keys.
_ORIGINAL_ST_PLOTLY_CHART = st.plotly_chart

def _safe_plotly_chart(fig, *args, **kwargs):
    if "key" not in kwargs or kwargs.get("key") is None:
        _n = st.session_state.get("_plotly_auto_key_counter", 0) + 1
        st.session_state["_plotly_auto_key_counter"] = _n
        title = "chart"
        try:
            title = str(fig.layout.title.text or "chart")[:40]
        except Exception:
            pass
        safe_title = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")[:40] or "chart"
        kwargs["key"] = f"auto_plotly_{safe_title}_{_n}"
    return _ORIGINAL_ST_PLOTLY_CHART(fig, *args, **kwargs)

st.plotly_chart = _safe_plotly_chart

# ============================================================
# PAGE CONFIG — must be the first Streamlit UI command
# ============================================================
st.set_page_config(
    page_title="ICCRE — India Climate Credit Risk Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": f"mailto:{CONTACT_EMAIL}",
        "Report a bug": f"mailto:{CONTACT_EMAIL}?subject=Bug Report",
        "About": (
            f"**ICCRE v{MODEL_VERSION}** — {PRODUCT_TAGLINE}\n\n"
            f"{PRODUCT_SUBTITLE}\n\n"
            f"Built on NGFS Phase III 2023 · ISSB IFRS S2 · RBI 2024 · SEBI BRSR Core\n\n"
            f"Contact: {CONTACT_EMAIL}"
        ),
    },
)


def _secret_or_env(name: str, default=None):
    """Return Streamlit secret first, then environment variable, then default."""
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def _get_groq_client():
    """Create Groq client from Streamlit Secrets or environment variable."""
    api_key = _secret_or_env("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Streamlit Secrets or environment variables.")
    return Groq(api_key=api_key)

# Public demo mode hides proprietary formulas, coefficients, and correction notes.
# Set ICCRE_INTERNAL_MODE="true" in Streamlit Secrets to expose internal methodology/calibration screens.
INTERNAL_MODE = str(_secret_or_env("ICCRE_INTERNAL_MODE", "false")).lower() in {"1", "true", "yes", "y"}
PUBLIC_MODE = not INTERNAL_MODE

# ============================================================
# GIS FILE LOADER — downloads large files from cloud storage
# on first run so they don't need to be in the GitHub repo
# ============================================================
GCS_BASE = _secret_or_env("GCS_BASE", "https://storage.googleapis.com/iccre-gis-data")  # ← override in Streamlit Secrets if needed

_GIS_FILES = {
    "Data/floodMapGL_rp100y.tif":        f"{GCS_BASE}/floodMapGL_rp100y.tif",
    "Data/ibtracs.NI.list.v04r01.csv":   f"{GCS_BASE}/ibtracs.NI.list.v04r01.csv",
    "Data/era5_test_day_grid.csv":        f"{GCS_BASE}/era5_test_day_grid.csv",
}

@st.cache_resource(show_spinner="⏳ Loading physical risk data (one-time, ~30s)...")
def _ensure_gis_files():
    os.makedirs("Data", exist_ok=True)
    failed = []
    for local_path, url in _GIS_FILES.items():
        if not os.path.exists(local_path):
            tmp_path = f"{local_path}.download"
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                urllib.request.urlretrieve(url, tmp_path)
                os.replace(tmp_path, local_path)
            except Exception as e:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                failed.append(f"{local_path}: {e}")
    return failed

# GIS files are intentionally NOT loaded at app startup.
# They are loaded only inside the Physical Risk module to avoid Streamlit Cloud health-check crashes.
if RASTERIO_IMPORT_ERROR:
    st.sidebar.warning("⚠️ Rasterio failed to import. Flood layer will be disabled, but other modules can still run.")
# ============================================================
# DESIGN SYSTEM  (UI-01)
# ============================================================
C = dict(
    bg_dark   = "#062F2E",
    bg_mid    = "#0B4D4B",
    bg_ocean  = "#115E6D",
    sidebar   = "#0A1F33",
    accent    = "#00F5D4",
    accent2   = "#22D3EE",
    accent3   = "#06B6D4",
    coral     = "#FF6B6B",
    amber     = "#FFD166",
    mint      = "#80ED99",
    purple    = "#A78BFA",
    white     = "#FFFFFF",
    off_white = "#E6F1F5",
    slate     = "#94A3B8",
    text      = "#E2E8F0",
    card      = "#0D3B4A",
)

def _chart_layout(title="", height=340, legend_override=None, margin_override=None, **kwargs):
    """
    Shared Plotly layout for all charts (UI-02).

    legend_override: dict merged into the default legend dict.
    margin_override: dict merged into the default margin dict.

    Use these instead of passing legend=/margin= directly to update_layout(),
    which would cause 'multiple values for keyword argument' TypeError.

    Example:
        fig.update_layout(**_chart_layout("Title",
            legend_override=dict(orientation="h", y=1.1),
            margin_override=dict(l=200, r=80)))
    """
    base_legend = dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["off_white"]))
    if legend_override:
        base_legend.update(legend_override)

    base_margin = dict(l=50, r=20, t=50, b=40)
    if margin_override:
        base_margin.update(margin_override)

    return dict(
        title=dict(text=title, font=dict(color=C["off_white"], size=14)),
        height=height,
        plot_bgcolor=C["bg_dark"],
        paper_bgcolor=C["bg_dark"],
        font=dict(color=C["off_white"], size=11),
        legend=base_legend,
        margin=base_margin,
        **kwargs,
    )

def _ax_style(fig, rows=1, cols=1):
    """Apply grid styling to all axes.
    Plain go.Figure (no make_subplots) must NOT receive row/col args —
    Plotly raises an Exception because there is no subplot grid.
    Only pass row/col when the figure was created with make_subplots().
    """
    if rows == 1 and cols == 1:
        # Plain figure — no row/col args
        fig.update_xaxes(gridcolor="#0B4D4B", zeroline=False)
        fig.update_yaxes(gridcolor="#0B4D4B", zeroline=False)
    else:
        # make_subplots figure — iterate all cells
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(gridcolor="#0B4D4B", zeroline=False, row=i, col=j)
                fig.update_yaxes(gridcolor="#0B4D4B", zeroline=False, row=i, col=j)
    return fig

# Scenario colour mapping — fixed per scenario name for consistency across ALL charts
SCENARIO_COLORS = {
    "Current Policies":                               C["coral"],   # red-coral
    "Nationally Determined Contributions (NDCs)":     C["amber"],   # amber-yellow
    "Below 2°C":                                      C["accent2"], # teal-cyan
    "Below 2C":                                       C["accent2"],
    "Below 2 Degrees":                                C["accent2"],
}

# Fallback palette for any extra / unrecognised scenarios — always distinguishable
_FALLBACK_PALETTE = [
    C["mint"],    # green
    C["purple"],  # purple
    "#F97316",    # orange
    "#EC4899",    # pink
    "#84CC16",    # lime
    "#14B8A6",    # teal-green
]
_fallback_idx: dict = {}   # scenario_name → palette index, populated on first use

def _scen_color(scen: str) -> str:
    """
    Return a stable, distinguishable colour for a scenario name.
    Named scenarios always get their fixed colour.
    Unknown scenarios cycle through _FALLBACK_PALETTE with a stable index
    (same scenario always gets same colour within one session).
    """
    global _fallback_idx
    # Direct match
    if scen in SCENARIO_COLORS:
        return SCENARIO_COLORS[scen]
    # Partial-match against known names (handles truncated scenario strings)
    scen_l = str(scen).lower()
    if "current" in scen_l and "polic" in scen_l:
        return C["coral"]
    if "ndc" in scen_l or "nationally determined" in scen_l:
        return C["amber"]
    if "below" in scen_l and "2" in scen_l:
        return C["accent2"]
    for k, v in SCENARIO_COLORS.items():
        if k in scen or scen in k:
            return v
    # Unknown scenario — assign a stable fallback colour
    if scen not in _fallback_idx:
        _fallback_idx[scen] = len(_fallback_idx) % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[_fallback_idx[scen]]

def _hex_rgba(hex_color: str, alpha: float) -> str:
    """
    Convert a 6-char hex color string to an rgba() string accepted by Plotly.
    Plotly does NOT accept 8-char hex (#RRGGBBAA) — must use rgba() notation.

    Example: _hex_rgba("#FF6B6B", 0.15) → "rgba(255,107,107,0.15)"
    """
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(6,182,212,{alpha})"   # safe fallback
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ============================================================
# PUBLIC UI HELPERS — result cards + output-scope labels
# ============================================================
def _fmt_money_cr(v, decimals=1):
    try:
        return f"₹{float(v):,.{decimals}f} Cr"
    except Exception:
        return "—"

def _fmt_pct(v, decimals=2):
    try:
        return f"{float(v):.{decimals}%}"
    except Exception:
        return "—"

def _fmt_num(v, suffix="", decimals=2):
    try:
        return f"{float(v):,.{decimals}f}{suffix}"
    except Exception:
        return "—"

def _fmt_capex_position(v):
    try:
        v=float(v)
        return f"₹{abs(v):,.0f} Cr {'Gap' if v>0 else 'Surplus' if v<0 else 'Balanced'}"
    except Exception:
        return "—"


def _metric_extreme(df, value_col, mode="max", scenario_col="Scenario", year_col="Year"):
    """Return value/year/scenario for peak or worst metric cards."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty or value_col not in df.columns:
            return None, None, None
        series = pd.to_numeric(df[value_col], errors="coerce")
        if series.dropna().empty:
            return None, None, None
        idx = series.idxmin() if mode == "min" else series.idxmax()
        row = df.loc[idx]
        val = float(row[value_col])
        yr = int(row[year_col]) if year_col in df.columns and pd.notna(row[year_col]) else None
        scen = str(row[scenario_col]) if scenario_col in df.columns else ""
        return val, yr, scen
    except Exception:
        return None, None, None

def _year_scenario_note(year, scenario, prefix="Achieved"):
    if year is None and not scenario:
        return ""
    if year is None:
        return f"{prefix} in scenario: {scenario}"
    if scenario:
        return f"{prefix} in {year} · Scenario: {scenario}"
    return f"{prefix} in {year}"


def scope_badge(scope: str, detail: str = ""):
    """Render a clear distinction between reporting-year and multi-year outputs."""
    if "single" in scope.lower() or "reporting" in scope.lower():
        label = f"📅 Reporting-Year Snapshot"
        color = C["mint"]
    elif "multi" in scope.lower() or "scenario" in scope.lower():
        label = f"📈 Multi-Year Scenario Projection"
        color = C["accent2"]
    else:
        label = scope
        color = C["amber"]
    extra = f" · {detail}" if detail else ""
    st.markdown(
        f"<div class='scope-badge' style='border-left:4px solid {color};'>"
        f"<span>{label}</span><small>{extra}</small></div>",
        unsafe_allow_html=True
    )

def metric_card(title, value, subtitle="", accent=None, scope=""):
    """Non-truncating KPI card replacing st.metric for market-facing result boxes."""
    accent = accent or C["accent2"]
    scope_html = f"<div class='metric-scope'>{scope}</div>" if scope else ""
    st.markdown(f"""
    <div class="metric-card-pro">
      <div class="metric-title-pro">{title}</div>
      <div class="metric-value-pro" style="color:{accent};">{value}</div>
      {f"<div class='metric-subtitle-pro'>{subtitle}</div>" if subtitle else ""}
      {scope_html}
    </div>
    """, unsafe_allow_html=True)

def render_metric_grid(items, columns=3):
    """items: list of dict(title, value, subtitle, accent, scope). Uses rows to avoid overlap."""
    if not items:
        return
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, item in zip(cols, items[i:i+columns]):
            with col:
                metric_card(
                    item.get("title", ""),
                    item.get("value", "—"),
                    item.get("subtitle", ""),
                    item.get("accent", C["accent2"]),
                    item.get("scope", ""),
                )

def clean_scenario_legend(fig, orientation="h"):
    """Move legends to a clean bottom position to avoid plot-title overlap."""
    fig.update_layout(
        legend=dict(
            orientation=orientation,
            yanchor="top",
            y=-0.18 if orientation == "h" else 1,
            xanchor="center" if orientation == "h" else "left",
            x=0.5 if orientation == "h" else 1.02,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color=C["off_white"]),
        ),
        margin=dict(l=55, r=30, t=70, b=95 if orientation == "h" else 45),
    )
    return fig


# ============================================================
# PERSISTENT DEMO OUTPUT RENDERERS
# These make tab outputs remain visible after tab switches/reruns.
# Results are cleared only by the sidebar Reset Demo Results button.
# ============================================================
def render_cached_transition_results():
    df_transition = st.session_state.get("df_transition")
    if not isinstance(df_transition, pd.DataFrame) or df_transition.empty:
        return False
    scope_badge("multi", "Saved transition results are retained until Reset Demo Results is clicked.")
    df_sum = (df_transition.groupby("Scenario").agg({
        "Carbon_Burden":"max","EBITDA_Margin":"min","DSCR":"min",
        "PD_Transition":"max","ECL_Transition":"max",
        "Stranded_Assets":"max","CAPEX_Gap":"max",
    }).round(4).reset_index())
    st.markdown(f"<h3 style='color:{C['white']};margin-top:16px;'>Scenario Risk Summary</h3>", unsafe_allow_html=True)
    def _color_pd(val):
        if isinstance(val, float):
            if val > 0.10: return f"color:{C['coral']};font-weight:600"
            if val > 0.05: return f"color:{C['amber']};font-weight:600"
            return f"color:{C['mint']};font-weight:600"
        return ""
    st.dataframe(
        df_sum.style
            .format({"Carbon_Burden":"{:.2%}","EBITDA_Margin":"{:.2%}","DSCR":"{:.2f}",
                     "PD_Transition":"{:.3%}","ECL_Transition":"{:.1f}",
                     "Stranded_Assets":"{:.0f}","CAPEX_Gap":"{:.0f}"})
            .map(_color_pd, subset=["PD_Transition"])
            .background_gradient(subset=["ECL_Transition"], cmap="Reds")
            .set_properties(**{"background-color":C["bg_dark"],"color":C["text"]}),
        width="stretch", hide_index=True
    )
    df_ry, actual_ry = reporting_year_slice(df_transition, REPORTING_YEAR)
    if isinstance(df_ry, pd.DataFrame) and not df_ry.empty:
        ry_worst = df_ry.loc[df_ry["PD_Transition"].idxmax()]
        scope_badge("single", f"Reporting-year snapshot: selected {REPORTING_YEAR}; displayed year {actual_ry}. Values show the worst selected scenario in that year.")
        render_metric_grid([
            {"title":f"{actual_ry} Worst-Scenario PD", "value":_fmt_pct(ry_worst['PD_Transition']), "subtitle":f"Scenario: {ry_worst['Scenario']}", "accent":C["accent2"], "scope":"Reporting year"},
            {"title":f"{actual_ry} Worst-Scenario ECL", "value":_fmt_money_cr(ry_worst['ECL_Transition']), "subtitle":"Credit loss at reporting-year stress", "accent":C["amber"], "scope":"Reporting year"},
            {"title":f"{actual_ry} DSCR", "value":_fmt_num(ry_worst['DSCR'], "×", 2), "subtitle":"Debt-service capacity in reporting year", "accent":C["coral"] if ry_worst['DSCR']<1.2 else C["mint"], "scope":"Reporting year"},
        ], columns=3)
    scope_badge("multi", "Multi-year scenario peak/worst values across all selected years.")
    _pd_v, _pd_y, _pd_s = _metric_extreme(df_transition, "PD_Transition", "max")
    _ecl_v, _ecl_y, _ecl_s = _metric_extreme(df_transition, "ECL_Transition", "max")
    _dscr_v, _dscr_y, _dscr_s = _metric_extreme(df_transition, "DSCR", "min")
    _str_v, _str_y, _str_s = _metric_extreme(df_transition, "Stranded_Assets", "max")
    render_metric_grid([
        {"title":"Peak Transition PD", "value":_fmt_pct(_pd_v), "subtitle":_year_scenario_note(_pd_y, _pd_s), "accent":C["accent2"], "scope":"Multi-year scenario"},
        {"title":"Peak ECL", "value":_fmt_money_cr(_ecl_v), "subtitle":_year_scenario_note(_ecl_y, _ecl_s), "accent":C["amber"], "scope":"Multi-year scenario"},
        {"title":"Minimum DSCR", "value":_fmt_num(_dscr_v, "×", 2), "subtitle":_year_scenario_note(_dscr_y, _dscr_s, prefix="Worst"), "accent":C["coral"] if (_dscr_v is not None and _dscr_v<1.2) else C["mint"], "scope":"Multi-year scenario"},
        {"title":"Peak Stranded Assets", "value":_fmt_money_cr(_str_v, 0), "subtitle":_year_scenario_note(_str_y, _str_s), "accent":C["purple"], "scope":"Multi-year scenario"},
    ], columns=4)
    fig_pd = go.Figure()
    for scen in df_transition["Scenario"].unique():
        ds = df_transition[df_transition["Scenario"] == scen]
        fig_pd.add_trace(go.Scatter(x=ds["Year"], y=ds["PD_Transition"], mode="lines+markers",
                                    name=scen[:30], line=dict(color=_scen_color(scen), width=2.5), marker=dict(size=6)))
    fig_pd.update_layout(**_chart_layout("PD Trajectory by NGFS Scenario", 320))
    fig_pd.update_yaxes(tickformat=".1%")
    _ax_style(fig_pd)
    st.plotly_chart(fig_pd, width="stretch")
    fig_ecl = go.Figure()
    for scen in df_transition["Scenario"].unique():
        ds = df_transition[df_transition["Scenario"] == scen]
        fig_ecl.add_trace(go.Bar(x=ds["Year"], y=ds["ECL_Transition"], name=scen[:30], marker_color=_scen_color(scen), opacity=0.85))
    fig_ecl.update_layout(**_chart_layout("Expected Credit Loss (₹ Cr) — All Scenarios", 300), barmode="group")
    _ax_style(fig_ecl)
    st.plotly_chart(fig_ecl, width="stretch")
    return True


def render_cached_physical_results():
    ps = st.session_state.get("phys_summary")
    df = st.session_state.get("phys_assets")
    df_phys_proj = st.session_state.get("df_physical_projection")
    if not isinstance(ps, dict) or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    scope_badge("single", f"Saved physical-risk results for reporting year {ps.get('Reporting Year', REPORTING_YEAR)}. Retained until Reset Demo Results is clicked.")
    render_metric_grid([
        {"title":"Total Revenue Loss", "value":_fmt_money_cr(ps.get('Total Revenue Loss (₹ Cr)',0)), "subtitle":"Estimated annual loss from asset downtime", "accent":C["amber"], "scope":"Saved reporting year"},
        {"title":"EBITDA Impact", "value":_fmt_money_cr(ps.get('EBITDA Loss (₹ Cr)',0)), "subtitle":"Revenue loss translated into EBITDA impact", "accent":C["coral"], "scope":"Saved reporting year"},
        {"title":"Post-Risk DSCR", "value":_fmt_num(ps.get('Post-Risk DSCR',0), "×", 2), "subtitle":"Debt service capacity after physical-risk shock", "accent":C["mint"] if float(ps.get('Post-Risk DSCR',0))>=1.2 else C["coral"], "scope":"Saved reporting year"},
        {"title":"Physical Risk PD", "value":_fmt_pct(ps.get('Physical Risk PD',0)), "subtitle":"Portfolio average physical-risk adjusted PD", "accent":C["accent2"], "scope":"Saved reporting year"},
        {"title":"ΔECL", "value":_fmt_money_cr(ps.get('ΔECL (₹ Cr)',0), 2), "subtitle":"Incremental expected credit loss", "accent":C["purple"], "scope":"Saved reporting year"},
        {"title":"Physical EAD Used", "value":_fmt_money_cr(ps.get('Effective Physical EAD (₹ Cr)',0), 1), "subtitle":str(ps.get('Physical EAD Allocation Mode','Demo allocation')), "accent":C["mint"], "scope":"Saved reporting year"},
    ], columns=3)
    st.subheader("🗺️ Asset Vulnerability Heatmap")
    score_map={"H_flood":"Flood","H_heat":"Heat","H_cyclone":"Cyclone"}
    cols=[c for c in score_map if c in df.columns]
    if cols:
        hm=df.set_index("asset_id")[cols].rename(columns={k:v for k,v in score_map.items() if k in cols})*100
        fig_hm=go.Figure(go.Heatmap(z=hm.values.T,x=hm.index.tolist(),y=hm.columns.tolist(),
            colorscale=[[0,C["mint"]],[0.3,C["amber"]],[0.6,"#F97316"],[1.0,C["coral"]]],
            text=[[f"{v:.0f}" for v in row] for row in hm.values.T],texttemplate="%{text}",
            colorbar=dict(title="Risk Score"),zmin=0,zmax=100))
        fig_hm.update_layout(**_chart_layout("Hazard Vulnerability Scores by Asset (0=none, 100=max)",max(220,len(df)*45+80)))
        _ax_style(fig_hm)
        st.plotly_chart(fig_hm,width="stretch")
    if isinstance(df_phys_proj, pd.DataFrame) and not df_phys_proj.empty:
        scope_badge("multi", "Saved forward-looking physical-risk projection across NGFS scenario years.")
        fig_pd_p = go.Figure()
        for scen in sorted(df_phys_proj["Scenario"].unique()):
            ds = df_phys_proj[df_phys_proj["Scenario"]==scen].sort_values("Year")
            fig_pd_p.add_trace(go.Scatter(x=ds["Year"], y=ds["PD_Physical"], mode="lines+markers", name=scen[:28],
                                           line=dict(color=_scen_color(scen), width=2.5), marker=dict(size=7)))
        fig_pd_p.update_layout(**_chart_layout("Physical PD — NGFS Scenario Projections", 280))
        fig_pd_p.update_yaxes(tickformat=".2%")
        _ax_style(fig_pd_p)
        st.plotly_chart(fig_pd_p, width="stretch")
    return True


def render_cached_brsr_results():
    bs = st.session_state.get("brsr_summary")
    bf = st.session_state.get("brsr_flags")
    if not isinstance(bs, pd.DataFrame) or bs.empty:
        return False
    row = bs.iloc[0]
    pd_adj = st.session_state.get("brsr_pd_adj", 0.0)
    risk_score = float(row.get("Overall_Risk_Score", 0))
    readiness = float(row.get("Readiness_Score", 0))
    sev_color = C["mint"] if risk_score < 30 else C["amber"] if risk_score < 60 else C["coral"]
    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {sev_color};border-radius:10px;padding:14px 18px;margin-bottom:12px;">
      <div style="font-size:16px;font-weight:700;color:{sev_color};">Saved BRSR Operational Climate Risk · Score: {risk_score:.0f}/100</div>
      <div style="font-size:13px;color:{C['slate']};margin-top:2px;">Governance signal: +{pd_adj*10000:.0f}bps · Readiness: {readiness:.0f}% · Results retained until Reset Demo Results is clicked.</div>
    </div>""", unsafe_allow_html=True)
    render_metric_grid([
        {"title":"Risk Score", "value":f"{risk_score:.0f}/100", "subtitle":"BRSR operational risk", "accent":sev_color, "scope":"Saved reporting year"},
        {"title":"BRSR Readiness", "value":f"{readiness:.0f}%", "subtitle":"SEBI BRSR Core readiness proxy", "accent":C["accent2"], "scope":"Saved reporting year"},
        {"title":"Governance Signal", "value":f"+{pd_adj*10000:.0f} bps", "subtitle":"Converted to bounded governance factor in integrated risk", "accent":C["purple"], "scope":"Saved reporting year"},
        {"title":"Water Cost Risk", "value":_fmt_money_cr(row.get('Water_Cost_Risk_Cr',0), 1), "subtitle":"5-year operational cost signal", "accent":C["amber"], "scope":"Saved reporting year"},
    ], columns=4)
    if isinstance(bf, pd.DataFrame) and not bf.empty:
        st.subheader("Active BRSR Flags")
        st.dataframe(bf, width="stretch", hide_index=True)
    return True


def render_cached_targets_results():
    eff = st.session_state.get("df_target_effect")
    df_tgt = st.session_state.get("df_target")
    if not isinstance(eff, pd.DataFrame) or eff.empty:
        return False
    scope_badge("multi", "Saved transition target scenario results are retained until Reset Demo Results is clicked.")
    st.markdown(f"<h4 style='color:{C['accent2']};'>Saved Financial Target Effectiveness</h4>", unsafe_allow_html=True)
    st.dataframe(eff.style.format({"PD_Target_Max":"{:.3%}","PD_Base_Max":"{:.3%}","PD_Reduction_%":"{:.1f}%"})
        .background_gradient(subset=["PD_Reduction_%"],cmap="Greens"), width="stretch", hide_index=True)
    df_base = st.session_state.get("df_transition")
    if isinstance(df_base, pd.DataFrame) and isinstance(df_tgt, pd.DataFrame):
        df_plot=df_base.merge(df_tgt,on=["Scenario","Year"],how="inner")
        fig_tgt=go.Figure()
        for scen in df_plot["Scenario"].unique():
            ds=df_plot[df_plot["Scenario"]==scen]; clr=_scen_color(scen)
            fig_tgt.add_trace(go.Scatter(x=ds["Year"],y=ds["PD_Transition"],mode="lines",name=f"{scen[:20]} Base",line=dict(dash="dash",color=clr,width=1.5)))
            fig_tgt.add_trace(go.Scatter(x=ds["Year"],y=ds["PD_Target"],mode="lines+markers",name=f"{scen[:20]} Target",line=dict(color=clr,width=2.5),marker=dict(size=6)))
        fig_tgt.update_layout(**_chart_layout("PD — Baseline vs Target", 340))
        fig_tgt.update_yaxes(tickformat=".1%")
        _ax_style(fig_tgt)
        st.plotly_chart(fig_tgt, width="stretch")
    return True

# ============================================================
# LOGGING
# ============================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def log_model_run(run_type, payload):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": ts, "model_version": MODEL_VERSION,
             "parameter_version": PARAMETER_VERSION, "ngfs_version": NGFS_DATA_VERSION,
             "engine": ENGINE_BUILD, "run_type": run_type, **payload}
    f = LOG_DIR / "run_log.csv"
    df = pd.DataFrame([entry])
    df.to_csv(f, mode="a", header=not f.exists(), index=False)

# ============================================================
# CSS  (UI-01)
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif !important;
}}

.stApp {{
    background: linear-gradient(145deg, {C["bg_dark"]} 0%, {C["bg_mid"]} 45%, {C["bg_ocean"]} 100%);
    background-attachment: fixed;
}}

/* ── SIDEBAR BASE ────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C["sidebar"]} 0%, #020617 100%) !important;
    border-right: 1px solid {C["bg_mid"]};
}}

/* All sidebar text */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown {{
    color: {C["text"]} !important;
}}

/* Sidebar widget labels */
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label {{
    color: {C["slate"]} !important;
    font-size: 12px !important;
}}

/* Sidebar input FIELDS — dark background, light text */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {{
    background-color: {C["card"]} !important;
    color: {C["accent2"]} !important;
    border: 1px solid {C["bg_ocean"]} !important;
    border-radius: 6px !important;
    caret-color: {C["accent2"]} !important;
}}
section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] textarea:focus {{
    border-color: {C["accent2"]} !important;
    box-shadow: 0 0 0 2px rgba(34,211,238,0.25) !important;
    outline: none !important;
}}
section[data-testid="stSidebar"] input::placeholder {{
    color: {C["slate"]} !important;
    opacity: 0.7;
}}

/* Sidebar SELECT boxes */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {{
    background-color: {C["card"]} !important;
    border: 1px solid {C["bg_ocean"]} !important;
    border-radius: 6px !important;
}}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] div {{
    color: {C["accent2"]} !important;
    background-color: transparent !important;
}}

/* Sidebar SELECT dropdown menu */
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {{
    background-color: {C["card"]} !important;
    border: 1px solid {C["bg_ocean"]} !important;
    border-radius: 8px !important;
}}
[data-baseweb="menu"] li,
[data-baseweb="option"] {{
    color: {C["text"]} !important;
    background-color: {C["card"]} !important;
}}
[data-baseweb="option"]:hover,
[data-baseweb="menu"] li:hover {{
    background-color: {C["bg_mid"]} !important;
    color: {C["accent2"]} !important;
}}

/* Sidebar number input spinner arrows */
section[data-testid="stSidebar"] .stNumberInput button {{
    background-color: {C["bg_mid"]} !important;
    border: 1px solid {C["bg_ocean"]} !important;
    color: {C["accent2"]} !important;
}}
section[data-testid="stSidebar"] .stNumberInput button:hover {{
    background-color: {C["accent3"]} !important;
    color: {C["bg_dark"]} !important;
}}

/* Sidebar SLIDER */
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {{
    background-color: {C["accent2"]} !important;
}}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {{
    color: {C["slate"]} !important;
}}

/* Sidebar CHECKBOX */
section[data-testid="stSidebar"] .stCheckbox span,
section[data-testid="stSidebar"] .stCheckbox label {{
    color: {C["text"]} !important;
}}

/* Sidebar divider */
section[data-testid="stSidebar"] hr {{
    border-color: {C["bg_mid"]} !important;
}}

/* Sidebar subheader / header */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    color: {C["accent2"]} !important;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
    margin-top: 12px !important;
}}

/* ── MAIN AREA TEXT ─────────────────────────────────────── */
h1,h2,h3,h4,h5,h6 {{ color: {C["white"]} !important; font-weight: 600 !important; }}
p, span, label, div {{ color: {C["text"]} !important; }}
.stMarkdown p {{ color: {C["text"]} !important; }}

/* Main area inputs also dark */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {{
    background-color: {C["card"]} !important;
    color: {C["accent2"]} !important;
    border: 1px solid {C["bg_ocean"]} !important;
    border-radius: 6px !important;
}}
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {{
    background-color: {C["card"]} !important;
    border: 1px solid {C["bg_ocean"]} !important;
    color: {C["accent2"]} !important;
}}

/* ── METRIC CARDS ───────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {C["card"]} 0%, {C["bg_mid"]} 100%);
    border: 1px solid {C["bg_ocean"]};
    border-radius: 10px;
    padding: 14px 16px !important;
}}
[data-testid="stMetricLabel"] {{ color: {C["slate"]} !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }}
[data-testid="stMetricValue"] {{ color: {C["accent2"]} !important; font-size: 22px !important; font-weight: 600 !important; font-family: 'IBM Plex Mono', monospace !important; }}
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

/* ── BUTTONS ────────────────────────────────────────────── */
.stButton > button {{
    background: linear-gradient(135deg, {C["accent3"]} 0%, {C["accent2"]} 100%) !important;
    color: {C["bg_dark"]} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 0.02em;
    padding: 8px 20px !important;
    transition: opacity 0.2s;
}}
.stButton > button:hover {{ opacity: 0.88; }}

/* ── TABS ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["bg_dark"]} !important;
    border-bottom: 1px solid {C["bg_mid"]};
    gap: 2px;
}}
.stTabs [data-baseweb="tab"] {{
    color: {C["slate"]} !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}}
.stTabs [aria-selected="true"] {{
    color: {C["accent2"]} !important;
    background: {C["card"]} !important;
    border-bottom: 2px solid {C["accent2"]} !important;
}}

/* ── DATAFRAMES ─────────────────────────────────────────── */
.stDataFrame {{ border-radius: 10px !important; overflow: hidden; }}
.stDataFrame [data-testid="stDataFrameResizable"] {{
    background: {C["bg_dark"]} !important;
    border: 1px solid {C["bg_mid"]} !important;
}}

/* ── ALERTS ─────────────────────────────────────────────── */
.stInfo    {{ background: rgba(6,182,212,0.1)   !important; border-left: 3px solid {C["accent3"]} !important; border-radius: 6px; }}
.stWarning {{ background: rgba(255,209,102,0.1) !important; border-left: 3px solid {C["amber"]}   !important; border-radius: 6px; }}
.stSuccess {{ background: rgba(128,237,153,0.1) !important; border-left: 3px solid {C["mint"]}    !important; border-radius: 6px; }}
.stError   {{ background: rgba(255,107,107,0.1) !important; border-left: 3px solid {C["coral"]}   !important; border-radius: 6px; }}

/* ── EXPANDER ───────────────────────────────────────────── */
.streamlit-expanderHeader {{ background: {C["card"]} !important; border-radius: 8px !important; color: {C["accent2"]} !important; }}

/* ── KPI CARD ───────────────────────────────────────────── */
.kpi-card {{
    background: linear-gradient(135deg, {C["card"]} 0%, {C["bg_mid"]} 100%);
    border: 1px solid {C["bg_ocean"]};
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}}
.kpi-value {{ font-size: 24px; font-weight: 600; color: {C["accent2"]}; font-family: 'IBM Plex Mono', monospace; }}
.kpi-label {{ font-size: 11px; color: {C["slate"]}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }}

/* ── RAG BADGES ─────────────────────────────────────────── */
.rag-green {{ color: {C["mint"]}  !important; font-weight: 600; }}
.rag-amber {{ color: {C["amber"]} !important; font-weight: 600; }}
.rag-red   {{ color: {C["coral"]} !important; font-weight: 600; }}

/* ── MISC ───────────────────────────────────────────────── */
hr {{ border-color: {C["bg_mid"]} !important; }}
code {{ background: {C["bg_dark"]} !important; color: {C["amber"]} !important; border-radius: 4px; font-family: 'IBM Plex Mono', monospace !important; }}

/* ── PUBLIC RESULT CARDS — no text truncation ───────────── */
.metric-card-pro {{
    background: linear-gradient(135deg, #0D3B4A 0%, #0B4D4B 100%);
    border: 1px solid #115E6D;
    border-radius: 12px;
    padding: 16px 14px;
    min-height: 126px;
    margin-bottom: 12px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 6px;
    overflow-wrap: anywhere;
    word-break: normal;
}}
.metric-title-pro {{
    color: #E6F1F5 !important;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
    white-space: normal !important;
    line-height: 1.25;
}}
.metric-value-pro {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: clamp(20px, 2.4vw, 30px);
    font-weight: 800;
    line-height: 1.08;
    white-space: normal !important;
}}
.metric-subtitle-pro {{
    color: #94A3B8 !important;
    font-size: 11px;
    line-height: 1.35;
    white-space: normal !important;
}}
.metric-scope {{
    margin-top: 4px;
    color: #80ED99 !important;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .04em;
    text-transform: uppercase;
}}
.scope-badge {{
    background: rgba(13,59,74,.85);
    border: 1px solid #115E6D;
    border-radius: 9px;
    padding: 10px 13px;
    margin: 12px 0 10px;
}}
.scope-badge span {{
    color: #FFFFFF !important;
    font-size: 13px;
    font-weight: 800;
}}
.scope-badge small {{
    color: #94A3B8 !important;
    font-size: 11px;
    margin-left: 8px;
}}

/* ── DEMO-1.1 SIDEBAR CONTRAST FIX ───────────────────────── */
section[data-testid="stSidebar"] .stButton > button,
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, #06B6D4 0%, #22D3EE 100%) !important;
    color: #062F2E !important;
    border: 1px solid #22D3EE !important;
    font-weight: 800 !important;
    text-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button *,
section[data-testid="stSidebar"] button[kind="secondary"] *,
section[data-testid="stSidebar"] button[kind="primary"] * {
    color: #062F2E !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] details,
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(13,59,74,0.95) !important;
    border: 1px solid #115E6D !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] summary,
section[data-testid="stSidebar"] summary *,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary * {
    color: #E6F1F5 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stExpander p,
section[data-testid="stSidebar"] [data-testid="stExpander"] p,
section[data-testid="stSidebar"] [data-testid="stExpander"] div {
    color: #E6F1F5 !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER BANNER
# ============================================================
st.markdown(f"""
<div style="background:linear-gradient(90deg,{C['bg_dark']} 0%,{C['bg_ocean']} 55%,{C['bg_mid']} 100%);
            border-bottom:2px solid {C['accent']};padding:18px 24px 14px;
            margin-bottom:4px;border-radius:0 0 12px 12px;">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
    <div>
      <div style="display:flex;align-items:center;gap:12px;">
        <div style="font-size:28px;font-weight:800;color:{C['white']};letter-spacing:-0.03em;
                    font-family:'IBM Plex Sans',sans-serif;">
          ICCRE
        </div>
        <div style="width:1px;height:32px;background:{C['bg_ocean']};"></div>
        <div>
          <div style="font-size:14px;font-weight:600;color:{C['accent2']};letter-spacing:0.01em;">
            {PRODUCT_TAGLINE}
          </div>
          <div style="font-size:11px;color:{C['slate']};margin-top:2px;">
            {PRODUCT_SUBTITLE}
          </div>
        </div>
      </div>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
      <span style="background:{C['card']};border:1px solid {C['bg_ocean']};color:{C['accent2']};
                   padding:4px 12px;border-radius:20px;font-size:11px;font-family:monospace;
                   font-weight:600;">
        v{MODEL_VERSION}
      </span>
      <span style="background:{C['card']};border:1px solid {C['bg_ocean']};color:{C['slate']};
                   padding:4px 12px;border-radius:20px;font-size:11px;">
        NGFS Phase III 2023
      </span>
      <a href="mailto:{CONTACT_EMAIL}" style="background:{C['accent']};color:{C['bg_dark']};
         padding:5px 14px;border-radius:20px;font-size:11px;font-weight:700;text-decoration:none;
         letter-spacing:0.02em;">
        ✉ Get Access
      </a>
    </div>
  </div>
  <div style="display:flex;gap:16px;margin-top:10px;flex-wrap:wrap;">
    <span style="font-size:10px;color:{C['slate']};background:{C['bg_dark']};
                 padding:3px 10px;border-radius:12px;border:1px solid {C['bg_mid']};">
      ✓ ISSB IFRS S2
    </span>
    <span style="font-size:10px;color:{C['slate']};background:{C['bg_dark']};
                 padding:3px 10px;border-radius:12px;border:1px solid {C['bg_mid']};">
      ✓ RBI Climate Guidelines 2024
    </span>
    <span style="font-size:10px;color:{C['slate']};background:{C['bg_dark']};
                 padding:3px 10px;border-radius:12px;border:1px solid {C['bg_mid']};">
      ✓ SEBI BRSR Core
    </span>
    <span style="font-size:10px;color:{C['slate']};background:{C['bg_dark']};
                 padding:3px 10px;border-radius:12px;border:1px solid {C['bg_mid']};">
      ✓ BCBS 2021
    </span>
    <span style="font-size:10px;color:{C['slate']};background:{C['bg_dark']};
                 padding:3px 10px;border-radius:12px;border:1px solid {C['bg_mid']};">
      ✓ ICAAP-ready
    </span>
  </div>
</div>
""", unsafe_allow_html=True)



# ============================================================
# PUBLIC DEMO ACCESS GATE — lead capture before access
# ============================================================
DEMO_ACCESS_FILE = Path("Data/demo_leads.csv")
PUBLIC_DEMO_LOCKED = True
PUBLIC_DEMO_NOTE = (
    "This public demo uses a fictional company. Inputs are locked so every user sees the same case. "
    "You can run and explore all output modules. Full company/portfolio analysis is available by pilot request."
)


def _valid_email(value: str) -> bool:
    value = (value or "").strip()
    return "@" in value and "." in value.split("@")[-1]


def _save_demo_lead(lead: dict):
    """Save user lead locally and optionally forward to a webhook."""
    os.makedirs("Data", exist_ok=True)
    row = {**lead, "captured_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}
    pd.DataFrame([row]).to_csv(DEMO_ACCESS_FILE, mode="a", header=not DEMO_ACCESS_FILE.exists(), index=False)

    webhook = _secret_or_env("LEADS_WEBHOOK_URL")
    if webhook:
        try:
            req = urllib.request.Request(
                webhook,
                data=json.dumps(row).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=4)
        except Exception:
            pass


def demo_access_gate():
    """Collect basic user details before opening the public demo."""
    if st.session_state.get("demo_access_granted"):
        return

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['card']},{C['bg_mid']});border:1px solid {C['bg_ocean']};border-radius:16px;padding:28px 34px;margin-top:18px;">
      <div style="font-size:24px;font-weight:800;color:{C['white']};margin-bottom:6px;">Access ICCRE Public Demo</div>
      <div style="font-size:13px;color:{C['slate']};line-height:1.7;max-width:900px;">
        {PUBLIC_DEMO_NOTE}<br>
        <b>Demo company:</b> Bharat Steel Industries Ltd · <b>Sector:</b> Steel · <b>Reporting year:</b> selectable after access
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("demo_lead_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name", placeholder="Your name")
        email = c2.text_input("Work Email *", placeholder="name@company.com")
        org = c1.text_input("Organisation *", placeholder="Company / institution")
        role = c2.selectbox("Role", ["Bank / Risk", "ESG / Sustainability", "Corporate Finance", "Consulting", "Investor / VC", "Founder / Startup", "Student / Researcher", "Other"])
        purpose = st.selectbox("Purpose of demo *", ["Evaluate for pilot", "Explore product", "Research / learning", "Partnership discussion", "Investment interest", "Other"])
        notes = st.text_area("Anything specific you want to evaluate?", placeholder="Optional", height=80)
        consent = st.checkbox("I agree that my details may be recorded for demo follow-up.")
        submitted = st.form_submit_button("Enter Demo", type="primary", width="stretch")

    if submitted:
        if not _valid_email(email):
            st.error("Please enter a valid email address.")
            st.stop()
        if not org.strip():
            st.error("Please enter your organisation.")
            st.stop()
        if not consent:
            st.error("Please confirm consent to continue.")
            st.stop()
        lead = {
            "name": name.strip(),
            "email": email.strip(),
            "organisation": org.strip(),
            "role": role,
            "purpose": purpose,
            "notes": notes.strip(),
            "demo_company": DEMO_DATASET["company_name"],
            "model_version": MODEL_VERSION,
        }
        _save_demo_lead(lead)
        st.session_state["demo_access_granted"] = True
        st.session_state["demo_lead"] = lead
        st.rerun()

    st.info("After access, open each tab and run the module you want to explore. Results stay saved until you click **Reset Demo Results**.")
    st.stop()


demo_access_gate()

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
_DEFAULTS = {
    "ai_outputs": {}, "suggested_params": {},
    "integrated_ran": False, "df_integrated_summary": None,
    "enable_transition": True, "enable_physical": False,
    "enable_targets": False, "enable_brsr": False,
    "transition_ran": False, "physical_ran": False,
    "targets_ran": False, "brsr_ran": False,
    "results_ready": False,
    "scope1": 0.0, "scope2": 0.0, "scope3": 0.0,
    "df_transition": None, "df_target": None,
    "phys_summary": None, "phys_assets": None, "df_physical_projection": None,
    "brsr_summary": None, "brsr_flags": None, "brsr_pd_adj": 0.0,
    "mc_results": None, "multi_year_results": None,
    "calibrated_params": None, "historical_data": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SECTOR PARAMETER REGISTRY
# ============================================================
# Public-ready alternative to scattered hard-coded constants:
# 1) All sector assumptions live in one versioned registry.
# 2) Values can be overridden with a zero-cost local file:
#       Data/sector_parameter_overrides.csv
#    columns: sector, parameter, value
# 3) Ranges and confidence labels are tracked for governance.
SECTOR_PARAMETER_REGISTRY = {
    "Steel": {
        "transition_physical_corr": 0.35, "gdp_sensitivity": 1.20, "carbon_demand": -0.25,
        "alpha_dscr": 1.10, "beta_carbon_credit": 1.40, "gamma_physical": 0.60, "delta_gdp": 0.80,
        "carbon_vol": 0.25, "physical_vol": 0.20, "gdp_vol": 0.05, "spread_sensitivity": 1.40,
        "ghg_p25": 250.0, "ghg_p50": 400.0, "ghg_p75": 650.0, "energy_p50": 125, "energy_p75": 180,
        "confidence": "Medium", "basis": "Sector-intensity expert benchmark; calibrate with borrower data when available."
    },
    "Power": {
        "transition_physical_corr": 0.30, "gdp_sensitivity": 0.70, "carbon_demand": -0.10,
        "alpha_dscr": 1.00, "beta_carbon_credit": 1.60, "gamma_physical": 0.50, "delta_gdp": 0.60,
        "carbon_vol": 0.22, "physical_vol": 0.18, "gdp_vol": 0.04, "spread_sensitivity": 1.20,
        "ghg_p25": 400.0, "ghg_p50": 700.0, "ghg_p75": 1100.0, "energy_p50": 85, "energy_p75": 130,
        "confidence": "Medium", "basis": "High carbon-price sensitivity with partial pass-through; calibrate locally."
    },
    "Cement": {
        "transition_physical_corr": 0.32, "gdp_sensitivity": 1.10, "carbon_demand": -0.20,
        "alpha_dscr": 1.05, "beta_carbon_credit": 1.20, "gamma_physical": 0.55, "delta_gdp": 0.70,
        "carbon_vol": 0.23, "physical_vol": 0.19, "gdp_vol": 0.05, "spread_sensitivity": 1.30,
        "ghg_p25": 200.0, "ghg_p50": 350.0, "ghg_p75": 550.0, "energy_p50": 110, "energy_p75": 160,
        "confidence": "Medium", "basis": "Energy/process-emission heavy sector; use plant-level calibration when available."
    },
    "Oil & Gas": {
        "transition_physical_corr": 0.40, "gdp_sensitivity": 0.90, "carbon_demand": -0.30,
        "alpha_dscr": 0.90, "beta_carbon_credit": 1.50, "gamma_physical": 0.65, "delta_gdp": 0.50,
        "carbon_vol": 0.30, "physical_vol": 0.22, "gdp_vol": 0.06, "spread_sensitivity": 1.50,
        "ghg_p25": 120.0, "ghg_p50": 220.0, "ghg_p75": 400.0, "energy_p50": 70, "energy_p75": 110,
        "confidence": "Medium", "basis": "Transition and stranded-asset sensitive; refine by business mix."
    },
    "Manufacturing": {
        "transition_physical_corr": 0.25, "gdp_sensitivity": 1.00, "carbon_demand": -0.15,
        "alpha_dscr": 1.00, "beta_carbon_credit": 0.80, "gamma_physical": 0.40, "delta_gdp": 0.90,
        "carbon_vol": 0.20, "physical_vol": 0.15, "gdp_vol": 0.04, "spread_sensitivity": 1.00,
        "ghg_p25": 50.0, "ghg_p50": 100.0, "ghg_p75": 200.0, "energy_p50": 60, "energy_p75": 95,
        "confidence": "Medium", "basis": "Generic diversified-manufacturing fallback; override for sub-sector."
    },
}

PARAMETER_BOUNDS = {
    "transition_physical_corr": (0.00, 0.80),
    "gdp_sensitivity": (0.20, 2.00),
    "carbon_demand": (-1.50, 0.20),
    "alpha_dscr": (0.20, 2.50),
    "beta_carbon_credit": (0.10, 3.00),
    "gamma_physical": (0.10, 2.00),
    "delta_gdp": (0.00, 2.00),
    "carbon_vol": (0.01, 0.80),
    "physical_vol": (0.01, 0.80),
    "gdp_vol": (0.005, 0.20),
    "spread_sensitivity": (0.10, 3.00),
}

@st.cache_data
def load_sector_parameter_registry():
    """Load local zero-cost overrides if provided; otherwise use embedded registry."""
    reg = {k: v.copy() for k, v in SECTOR_PARAMETER_REGISTRY.items()}
    override_paths = [Path("Data/sector_parameter_overrides.csv"), Path("data/sector_parameter_overrides.csv")]
    for op in override_paths:
        if op.exists():
            try:
                odf = pd.read_csv(op)
                for _, row in odf.iterrows():
                    sec = str(row.get("sector", "")).strip()
                    par = str(row.get("parameter", "")).strip()
                    if sec in reg and par in reg[sec]:
                        val = float(row.get("value"))
                        if par in PARAMETER_BOUNDS:
                            lo, hi = PARAMETER_BOUNDS[par]
                            val = float(np.clip(val, lo, hi))
                        reg[sec][par] = val
                        reg[sec]["basis"] = "Local override file: Data/sector_parameter_overrides.csv"
            except Exception:
                pass
            break
    return reg

_SECTOR_REG = load_sector_parameter_registry()

def _reg_value(sector_name, key, fallback=None):
    return _SECTOR_REG.get(sector_name, _SECTOR_REG["Manufacturing"]).get(key, fallback)

SECTOR_CORRELATION = {s: v["transition_physical_corr"] for s, v in _SECTOR_REG.items()}
SECTOR_GDP_SENSITIVITY = {s: v["gdp_sensitivity"] for s, v in _SECTOR_REG.items()}
SECTOR_CARBON_DEMAND = {s: v["carbon_demand"] for s, v in _SECTOR_REG.items()}
SECTOR_CREDIT_PARAMS = {
    s: {
        "alpha_dscr": v["alpha_dscr"],
        "beta_carbon_credit": v["beta_carbon_credit"],
        "gamma_physical": v["gamma_physical"],
        "delta_gdp": v["delta_gdp"],
    }
    for s, v in _SECTOR_REG.items()
}
SECTOR_MC_PARAMS = {
    s: {"carbon_vol": v["carbon_vol"], "physical_vol": v["physical_vol"], "gdp_vol": v["gdp_vol"]}
    for s, v in _SECTOR_REG.items()
}
SECTOR_SPREAD_SENSITIVITY = {s: v["spread_sensitivity"] for s, v in _SECTOR_REG.items()}

# Kept as internal model-support structures; not exposed in public UI.
SECTOR_PORTFOLIO_CORR = {
    "Steel":         {"Steel": 0.35, "Power": 0.25, "Cement": 0.30, "Oil & Gas": 0.28, "Manufacturing": 0.22},
    "Power":         {"Steel": 0.25, "Power": 0.40, "Cement": 0.27, "Oil & Gas": 0.35, "Manufacturing": 0.20},
    "Cement":        {"Steel": 0.30, "Power": 0.27, "Cement": 0.38, "Oil & Gas": 0.26, "Manufacturing": 0.24},
    "Oil & Gas":     {"Steel": 0.28, "Power": 0.35, "Cement": 0.26, "Oil & Gas": 0.45, "Manufacturing": 0.22},
    "Manufacturing": {"Steel": 0.22, "Power": 0.20, "Cement": 0.24, "Oil & Gas": 0.22, "Manufacturing": 0.30},
}
SECTOR_CONTAGION = {
    "Power":     {"Steel": 0.20, "Cement": 0.15},
    "Steel":     {"Manufacturing": 0.18, "Cement": 0.10},
    "Oil & Gas": {"Power": 0.22, "Manufacturing": 0.15},
}
SCENARIO_WEIGHTS = {"Current Policies": 0.50, "Nationally Determined Contributions (NDCs)": 0.30, "Below 2°C": 0.20}
# Zero-cost production-readiness overlays. These are transparent management overlays,
# not externally purchased/calibrated datasets.
BRSR_MAX_GOVERNANCE_MULTIPLIER = 1.20
SCENARIO_WEIGHTING_METHOD = "Management-weighted NGFS ECL"
MODEL_LIMITATION_NOTE = (
    "Sector parameters are versioned assumptions and can be overridden with client data. "
    "For regulated lending/provisioning, calibrate against borrower/default history."
)
SCENARIO_REGISTRY = {
    "Current Policies": {"type": "Baseline", "temperature": "≈2.7°C", "policy_intensity": "Low"},
    "Nationally Determined Contributions (NDCs)": {"type": "Policy Transition", "temperature": "≈2.1°C", "policy_intensity": "Medium"},
    "Below 2°C": {"type": "Orderly / lower-temperature transition", "temperature": "≈1.7–2.0°C", "policy_intensity": "High"},
}
SECTOR_GHG_BENCHMARKS = {
    s: {"p25": v["ghg_p25"], "p50": v["ghg_p50"], "p75": v["ghg_p75"]}
    for s, v in _SECTOR_REG.items()
}
SECTOR_ENERGY_BENCHMARKS = {
    s: {"p50": v["energy_p50"], "p75": v["energy_p75"]}
    for s, v in _SECTOR_REG.items()
}
BRSR_PD_SPREAD = {
    "High GHG intensity (>P75)": 0.0035, "Low renewable energy (<20%)": 0.0018,
    "High energy intensity (>P75)": 0.0022, "High water stress + high withdrawal": 0.0028,
    "Low target coverage (<50%)": 0.0030, "High hazardous waste (>30%)": 0.0012,
    "No verified emissions data": 0.0025, "Missing Scope 3 disclosure": 0.0010,
}

PD_FLOOR = 0.0005
PD_CAP   = 0.35
LGD_PHYSICAL_MULTIPLIER = 0.30
BASE_CREDIT_SPREAD = 0.02
P_START = 50; P_FULL = 250; MAX_STRANDING = 0.80
margin_erosion_rate = 0.005; margin_floor = 0.10
SYSTEMIC_CLIMATE_WEIGHT = 0.50; SECTOR_SHOCK_WEIGHT = 0.25
CONTAGION_WEIGHT = 0.15; IDIOSYNCRATIC_WEIGHT = 0.10
CLIMATE_DRIVER_CORR = np.array([[1.0, 0.30, -0.40],[0.30, 1.0, -0.20],[-0.40, -0.20, 1.0]])

# ============================================================
# MATH HELPERS
# ============================================================
def logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    v = np.log(p / (1 - p))
    return v if np.isfinite(v) else 0.0

def sigmoid(x):
    x = np.clip(x, -50, 50)
    v = 1 / (1 + np.exp(-x))
    return v if np.isfinite(v) else 0.0

def gaussian_copula_pd(pd_t, pd_p, rho):
    pd_t = np.clip(pd_t, 1e-6, 1-1e-6)
    pd_p = np.clip(pd_p, 1e-6, 1-1e-6)
    z_t  = norm.ppf(pd_t); z_p = norm.ppf(pd_p)
    return float(np.clip(multivariate_normal.cdf([z_t, z_p], [0,0], [[1,rho],[rho,1]]), 0, 1))

def combined_pd_union(pd_t, pd_p, rho):
    """Combined default probability using union logic.
    Gaussian copula gives the joint intersection probability P(T ∩ P).
    Integrated borrower PD should represent P(T ∪ P), not only the intersection.
    """
    pd_t = float(np.clip(pd_t or 0.0, PD_FLOOR, PD_CAP))
    pd_p = float(np.clip(pd_p or 0.0, 0.0, PD_CAP))
    if pd_p <= 0:
        return pd_t, 0.0
    joint_intersection = gaussian_copula_pd(pd_t, pd_p, rho)
    pd_union = pd_t + pd_p - joint_intersection
    return float(np.clip(pd_union, max(pd_t, pd_p, PD_FLOOR), PD_CAP)), float(joint_intersection)

def brsr_governance_multiplier(brsr_pd_adj, max_multiplier=1.20):
    """Convert BRSR basis-point signal into a bounded governance multiplier.
    Avoids adding BRSR as a third independent default block and reduces double counting.
    """
    if brsr_pd_adj is None or brsr_pd_adj <= 0:
        return 1.0
    return float(np.clip(1.0 + (float(brsr_pd_adj) / 0.015) * (max_multiplier - 1.0), 1.0, max_multiplier))

def ecl_cr(pd, lgd, ead_cr):
    """Expected credit loss in ₹ Cr. EAD inputs in this app are already ₹ Cr, so no /1000 conversion."""
    return float(np.clip(pd, 0, 1) * np.clip(lgd, 0, 1) * max(float(ead_cr), 0.0))

def scenario_weighted_peak(df, metric, scenario_col="Scenario"):
    if not isinstance(df, pd.DataFrame) or df.empty or metric not in df.columns or scenario_col not in df.columns:
        return None
    grp = df.groupby(scenario_col)[metric].max()
    weights = {s: SCENARIO_WEIGHTS.get(s, 1.0) for s in grp.index}
    total = sum(weights.values()) or 1.0
    return float(sum(grp[s] * weights[s] / total for s in grp.index))

def simulate_carbon_price_path(start, years, drift=0.05, vol=0.25):
    prices = [start]
    for _ in range(years):
        prices.append(prices[-1] * np.exp((drift - 0.5*vol**2) + vol*np.random.normal()))
    return prices

def haversine(lat1, lon1, lat2, lon2):
    ln1,lt1,ln2,lt2 = map(radians,[lon1,lat1,lon2,lat2])
    a = sin((lt2-lt1)/2)**2 + cos(lt1)*cos(lt2)*sin((ln2-ln1)/2)**2
    return 6371*2*asin(sqrt(a))

# Damage curves
def flood_damage(d):
    if d<=0: return 0.0
    if d<0.3: return 0.02
    if d<0.7: return 0.10
    if d<1.5: return 0.35
    if d<2.5: return 0.60
    return 0.85

def heat_damage(d):
    if d<5: return 0.01
    if d<15: return 0.08
    if d<30: return 0.20
    if d<50: return 0.40
    return 0.65

def cyclone_damage(w):
    if w<80: return 0.02
    if w<120: return 0.15
    if w<160: return 0.35
    if w<200: return 0.60
    return 0.85

# ============================================================
# NGFS DATA
# ============================================================
@st.cache_data
def load_ngfs():
    candidates = [
        Path("Data/ngfs_scenarios.csv"),
        Path("data/ngfs_scenarios.csv"),
        Path("ngfs_scenarios.csv"),
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.replace("﻿", "", regex=False)
            return df
    st.error("❌ File not found: Data/ngfs_scenarios.csv. Upload it to GitHub exactly at Data/ngfs_scenarios.csv.")
    st.stop()

df_ngfs = load_ngfs()
for c in df_ngfs.select_dtypes("object"):
    df_ngfs[c] = df_ngfs[c].astype(str).str.strip()
keywords = ["Price|Carbon","GDP|MER","AR6 climate diagnostics|Surface Temperature"]
df_ngfs  = df_ngfs[df_ngfs["Variable"].apply(lambda x: any(k.lower() in x.lower() for k in keywords))]
year_cols = [c for c in df_ngfs.columns if c.isdigit()]
df_long   = df_ngfs.melt(
    id_vars=[c for c in df_ngfs.columns if c not in year_cols],
    value_vars=year_cols, var_name="Year", value_name="Value"
)
df_long["Year"] = df_long["Year"].astype(int)

# ============================================================
# PUBLIC DEMO SCENARIO FILTER
# ============================================================
# Public demo intentionally exposes only three scenario families:
# 1) Current Policies, 2) NDCs, 3) Below 2°C.
# The NGFS source file may use slightly different labels, so we resolve aliases robustly.
PUBLIC_SCENARIO_FAMILIES = {
    "Current Policies": ["current policies"],
    "Nationally Determined Contributions (NDCs)": ["nationally determined contributions", "ndc"],
    "Below 2°C": ["below 2", "below 2°", "below 2c", "below 2 c", "below 2 degree"],
}

def _match_public_scenario_name(name: str, aliases: list[str]) -> bool:
    n = str(name).lower().replace("°", "").replace("degrees", "degree")
    for alias in aliases:
        a = alias.lower().replace("°", "").replace("degrees", "degree")
        if a in n:
            return True
    return False

def get_public_demo_scenarios(df):
    available = sorted(df["Scenario"].dropna().astype(str).unique())
    selected = []
    for display, aliases in PUBLIC_SCENARIO_FAMILIES.items():
        match = next((sc for sc in available if _match_public_scenario_name(sc, aliases)), None)
        if match and match not in selected:
            selected.append(match)
    return selected

PUBLIC_DEMO_SCENARIOS = get_public_demo_scenarios(df_long)
if PUBLIC_DEMO_SCENARIOS:
    df_long = df_long[df_long["Scenario"].isin(PUBLIC_DEMO_SCENARIOS)].copy()

def _nearest_available_year(years, target_year):
    vals = sorted(int(y) for y in set(years))
    if not vals:
        return int(target_year)
    return min(vals, key=lambda yy: abs(yy-int(target_year)))

def reporting_year_slice(df, reporting_year, year_col="Year"):
    """Return rows for selected reporting year; if not available, use nearest scenario year."""
    if not isinstance(df, pd.DataFrame) or df.empty or year_col not in df.columns:
        return df, reporting_year
    actual_year = _nearest_available_year(df[year_col].dropna().astype(int).unique(), int(reporting_year))
    return df[df[year_col].astype(int) == int(actual_year)].copy(), actual_year

# ============================================================
# AI PARAMETER CALIBRATION
# ============================================================
def ai_calibrate_parameters(sector, *_):
    presets = {
        "Steel":         {"carbon_pass_through":0.35,"demand_elasticity":-0.40,"price_elasticity":-0.10,"beta_carbon_transition":1.35,"abatement_potential":0.30},
        "Power":         {"carbon_pass_through":0.65,"demand_elasticity":-0.15,"price_elasticity":-0.02,"beta_carbon_transition":1.50,"abatement_potential":0.45},
        "Cement":        {"carbon_pass_through":0.40,"demand_elasticity":-0.25,"price_elasticity":-0.05,"beta_carbon_transition":1.25,"abatement_potential":0.35},
        "Oil & Gas":     {"carbon_pass_through":0.55,"demand_elasticity":-0.35,"price_elasticity":-0.08,"beta_carbon_transition":1.45,"abatement_potential":0.25},
        "Manufacturing": {"carbon_pass_through":0.50,"demand_elasticity":-0.20,"price_elasticity":-0.04,"beta_carbon_transition":0.90,"abatement_potential":0.40},
    }
    return presets.get(sector, presets["Manufacturing"])


# ============================================================
# LOCKED PUBLIC DEMO SIDEBAR — fixed fictional company, all modules available
# ============================================================
st.sidebar.markdown(f"""
<div style="padding:14px 0 10px;">
  <div style="font-size:20px;font-weight:800;color:{C['accent2']};letter-spacing:-0.02em;">ICCRE</div>
  <div style="font-size:10px;color:{C['slate']};margin-top:1px;line-height:1.5;">
    Locked public demo<br>
    <span style="color:{C['bg_ocean']}">v{MODEL_VERSION} · {MODEL_BUILD_DATE}</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style='background:{C['amber']}22;border:1px solid {C['amber']};border-radius:8px;padding:10px 12px;margin:8px 0;font-size:11px;line-height:1.55;'>
  <b style='color:{C['amber']};'>🔒 Demo company locked</b><br>
  Bharat Steel Industries Ltd · Steel · Reporting Year selectable<br>
  Company inputs cannot be changed in this public demo.
</div>
""", unsafe_allow_html=True)

lead = st.session_state.get("demo_lead", {})
if lead:
    st.sidebar.caption(f"Access: {lead.get('email','')}")

clear_demo = st.sidebar.button("↻ Reset Demo Results", width="stretch", help="Clears all saved demo outputs. Results otherwise remain available while you switch tabs.")
if clear_demo:
    _RESET_KEYS = [
        "transition_ran","physical_ran","targets_ran","brsr_ran","integrated_ran",
        "results_ready","df_transition","df_transition_summary","df_target",
        "df_target_effect","phys_summary","phys_assets","df_physical_projection",
        "brsr_summary","brsr_flags","brsr_pd_adj","brsr_target_overlay",
        "brsr_pd_reduction","brsr_remaining_flags","df_integrated_summary",
        "mc_results","multi_year_results","calibrated_params",
        "historical_data","validation_results","ai_outputs","run_full_demo_requested",
    ]
    for _k in _RESET_KEYS:
        if _k in st.session_state:
            del st.session_state[_k]
    for _k2, _v2 in _DEFAULTS.items():
        st.session_state[_k2] = _v2
    st.rerun()

with st.sidebar.expander("📋 How to use this demo", expanded=True):
    st.markdown(f"""
<div style="font-size:11px;color:{C['off_white']};line-height:1.8;">
1. Run the module tab you want to explore<br>
2. Open <b>Dashboard</b> for board-level summary<br>
3. Explore <b>Transition</b>, <b>Physical</b>, <b>BRSR</b>, <b>Targets</b>, and <b>Integrated Risk</b><br>
4. Results remain saved while switching tabs<br>
5. Use <b>Reset Demo Results</b> only when you want to clear outputs
</div>
""", unsafe_allow_html=True)

# Fixed demo inputs — no public company input fields.
for k, v in DEMO_DATASET.items():
    st.session_state[f"_demo_{k}"] = v
st.session_state["_demo_loaded"] = True

company_name = DEMO_DATASET["company_name"]
sector = DEMO_DATASET["sector"]

_available_years = sorted(int(y) for y in df_long["Year"].dropna().unique())
_default_reporting_year = int(DEMO_DATASET["reporting_year"])
_default_reporting_year = _nearest_available_year(_available_years, _default_reporting_year)
REPORTING_YEAR = int(st.sidebar.selectbox(
    "Reporting Year",
    options=_available_years,
    index=_available_years.index(_default_reporting_year) if _default_reporting_year in _available_years else 0,
    help="Select the reporting year for snapshot outputs. Scenario charts still show the full multi-year pathway.",
    key="selected_reporting_year",
))
BASELINE_SCENARIO = "Current Policies"
default_correlation = SECTOR_CORRELATION.get(sector, 0.30)

revenue_0           = float(DEMO_DATASET["revenue_0"])
ebitda_margin_0     = float(DEMO_DATASET["ebitda_margin_0"])
interest_payment    = float(DEMO_DATASET["interest_payment"])
total_assets        = float(DEMO_DATASET["total_assets"])
exposure_at_default = float(DEMO_DATASET["exposure_at_default"])

scope1 = float(DEMO_DATASET["scope1"])
scope2 = float(DEMO_DATASET["scope2"])
scope3 = float(DEMO_DATASET["scope3"])
TOTAL_EMISSIONS = scope1 + scope2 + scope3
st.session_state["scope1"] = scope1
st.session_state["scope2"] = scope2
st.session_state["scope3"] = scope3
high_carbon_assets = float(DEMO_DATASET["high_carbon_assets"])

# All major modules are enabled for the locked demo.
st.session_state["enable_transition"] = True
st.session_state["enable_physical"]   = True
st.session_state["enable_targets"]    = True
st.session_state["enable_brsr"]       = True

USD_INR = 83.0
US_CPI_2010_TO_2026 = 1.38
CARBON_PRICE_INFLATION_FACTOR = US_CPI_2010_TO_2026
carbon_pass_through = float(DEMO_DATASET["carbon_pass_through"])
demand_elasticity = float(DEMO_DATASET["demand_elasticity"])
price_elasticity = -0.10
beta_carbon_transition = 1.35
planned_capex = float(DEMO_DATASET["planned_capex"])
abatement_cost = float(DEMO_DATASET["abatement_cost"])
abatement_potential = float(DEMO_DATASET["abatement_potential"])
base_pd = float(DEMO_DATASET["base_pd"])
LGD_0 = float(DEMO_DATASET["lgd_0"])
tax_rate = 0.25
G = np.mean([0.7, 0.5, 0.4, 0.6])

sector_params      = SECTOR_CREDIT_PARAMS.get(sector,SECTOR_CREDIT_PARAMS["Manufacturing"])
alpha_dscr         = sector_params["alpha_dscr"]
beta_carbon_credit = sector_params["beta_carbon_credit"]
gamma_phys         = sector_params["gamma_physical"]
delta_gdp          = sector_params["delta_gdp"]
spread_beta        = SECTOR_SPREAD_SENSITIVITY.get(sector,1.0)
gdp_sensitivity    = SECTOR_GDP_SENSITIVITY.get(sector,1.0)
mc_params          = SECTOR_MC_PARAMS.get(sector,SECTOR_MC_PARAMS["Manufacturing"])
MC_CARBON_VOL=mc_params["carbon_vol"]; MC_PHYS_VOL=mc_params["physical_vol"]; MC_GDP_VOL=mc_params["gdp_vol"]
CLIMATE_DRIVER_VOL = np.array([MC_CARBON_VOL,MC_PHYS_VOL,MC_GDP_VOL])
CLIMATE_DRIVER_COV = np.outer(CLIMATE_DRIVER_VOL,CLIMATE_DRIVER_VOL)*CLIMATE_DRIVER_CORR

st.sidebar.markdown(f"""
<div style='background:{C['card']};border:1px solid {C['bg_ocean']};border-radius:8px;padding:10px 12px;margin-top:12px;font-size:10px;line-height:1.7;color:{C['slate']};'>
<b style='color:{C['accent2']};'>Demo inputs</b><br>
Reporting Year: {REPORTING_YEAR}<br>
Revenue: ₹{revenue_0:,.0f} Cr<br>
EAD: ₹{exposure_at_default:,.0f} Cr<br>
Emissions: {TOTAL_EMISSIONS/1e6:.1f} Mn tCO₂e<br>
Base PD: {base_pd:.2%} · LGD: {LGD_0:.0%}
</div>
""", unsafe_allow_html=True)

# ============================================================
# PUBLIC DEMO EXPORT — summary only, full report by request
# ============================================================
st.sidebar.divider()
st.sidebar.markdown(f"<div style='font-size:12px;font-weight:600;color:{C['accent2']};text-transform:uppercase;letter-spacing:.06em;'>📥 Demo Export</div>", unsafe_allow_html=True)

any_ran = any(st.session_state.get(k, False) for k in ["transition_ran","physical_ran","brsr_ran","integrated_ran"])
if any_ran:
    import json as _json
    _export_payload = {
        "metadata": {
            "model_version": MODEL_VERSION,
            "company": company_name,
            "sector": sector,
            "reporting_year": REPORTING_YEAR,
            "demo_note": "Public demo; fictional company; inputs locked.",
            "timestamp": datetime.utcnow().isoformat(),
        },
        "reporting_year_snapshot": {},
        "multi_year_scenario_results": {},
        "model_use_note": MODEL_USE_NOTE,
    }
    if isinstance(st.session_state.get("df_transition"), pd.DataFrame) and not st.session_state["df_transition"].empty:
        _dt = st.session_state["df_transition"]
        _dt_ry, _actual_ry = reporting_year_slice(_dt, REPORTING_YEAR)
        if isinstance(_dt_ry, pd.DataFrame) and not _dt_ry.empty:
            _ry_worst = _dt_ry.loc[_dt_ry["PD_Transition"].idxmax()]
            _export_payload["reporting_year_snapshot"].update({
                "displayed_year": int(_actual_ry),
                "worst_reporting_year_scenario": str(_ry_worst["Scenario"]),
                "transition_pd": float(_ry_worst["PD_Transition"]),
                "transition_ecl_cr": float(_ry_worst["ECL_Transition"]),
                "dscr": float(_ry_worst["DSCR"]),
                "carbon_burden": float(_ry_worst["Carbon_Burden"]),
            })
        _export_payload["multi_year_scenario_results"].update({
            "peak_transition_pd": float(_dt["PD_Transition"].max()),
            "peak_transition_ecl_cr": float(_dt["ECL_Transition"].max()),
            "worst_dscr": float(_dt["DSCR"].min()),
        })
    if isinstance(st.session_state.get("phys_summary"), dict):
        _export_payload["reporting_year_snapshot"].update(st.session_state["phys_summary"])
    if st.session_state.get("brsr_ran"):
        _export_payload["reporting_year_snapshot"]["brsr_governance_signal_bps"] = float(st.session_state.get("brsr_pd_adj",0)*10000)
    if isinstance(st.session_state.get("df_integrated_summary"), pd.DataFrame):
        _export_payload["integrated_risk"] = st.session_state["df_integrated_summary"].to_dict("records")

    st.sidebar.download_button(
        label="⬇️ Download Demo Summary (JSON)",
        data=_json.dumps(_export_payload, indent=2, default=str).encode(),
        file_name=f"ICCRE_demo_summary_{REPORTING_YEAR}.json",
        mime="application/json",
        width="stretch",
        help="Public demo summary only. Full Excel/model reports are available for pilot users.",
    )
else:
    st.sidebar.caption("Run the demo to enable summary export.")

st.sidebar.markdown(f"""
<div style="background:{C['card']};border:1px solid {C['bg_ocean']};border-radius:8px;padding:12px 14px;margin-top:10px;">
  <div style="font-size:11px;font-weight:700;color:{C['accent2']};text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">Want the full report?</div>
  <div style="font-size:10px;color:{C['slate']};margin-bottom:8px;line-height:1.6;">
    Full Excel reports, custom company inputs, calibration, portfolio analysis and model documentation are available for serious pilot users.
  </div>
  <a href="mailto:{CONTACT_EMAIL}?subject=ICCRE Pilot Access Request"
     style="display:block;background:{C['accent']};color:{C['bg_dark']};text-align:center;padding:6px 0;border-radius:6px;font-size:11px;font-weight:700;text-decoration:none;letter-spacing:.02em;">
    ✉ Request Pilot Access
  </a>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TRANSITION RISK ENGINE (v1.1 all fixes preserved)
# ============================================================
def run_transition_engine(df_long, selected_scenarios, revenue_0, ebitda_margin_0,
        interest_payment, TOTAL_EMISSIONS, high_carbon_assets, exposure_at_default,
        carbon_pass_through, demand_elasticity, price_elasticity, beta_carbon_transition,
        beta_carbon_credit, USD_INR, G, base_pd, LGD_0, abatement_cost,
        abatement_potential, planned_capex, P_START, P_FULL, MAX_STRANDING,
        margin_erosion_rate, margin_floor, alpha_dscr, tax_rate,
        CARBON_PRICE_INFLATION_FACTOR, baseline_temp=1.5, temp_sensitivity=0.02,
        gamma_phys=0.5, delta_gdp=0.8):
    results = []
    for scen in selected_scenarios:
        d      = df_long[df_long["Scenario"]==scen]
        d_base = df_long[df_long["Scenario"]==BASELINE_SCENARIO]
        for y in sorted(d["Year"].unique()):
            try:
                carbon_price = d.loc[(d["Year"]==y)&d["Variable"].str.contains("Carbon",case=False),"Value"].iloc[0]
                temp         = d.loc[(d["Year"]==y)&d["Variable"].str.contains("Temperature",case=False),"Value"].iloc[0]
                gdp          = d.loc[(d["Year"]==y)&d["Variable"].str.contains("GDP",case=False),"Value"].iloc[0]
            except Exception:
                continue
            try:
                gdp_base  = d_base.loc[(d_base["Year"]==y)&d_base["Variable"].str.contains("GDP",case=False),"Value"].iloc[0]
                gdp_shock = (gdp-gdp_base)/max(gdp_base,1e-6)
            except:
                gdp_shock = 0.0
            # FIX-05: two-step CPI conversion
            carbon_price_adj  = carbon_price * CARBON_PRICE_INFLATION_FACTOR
            gross_carbon_cost = TOTAL_EMISSIONS * carbon_price_adj * USD_INR / 1e7
            net_carbon_cost   = gross_carbon_cost * (1-carbon_pass_through)
            # Revenue (FIX-03, FIX-06)
            revenue = revenue_0*(1+gdp_sensitivity*gdp_shock)
            revenue *= (1+demand_elasticity*(net_carbon_cost/max(revenue_0,1)))
            revenue *= (1+price_elasticity*(net_carbon_cost/max(revenue_0,1)))  # FIX-03
            physical_loss = temp_sensitivity*max(0,temp-baseline_temp)
            revenue *= (1-physical_loss)
            revenue_floor = max(revenue_0 * 0.05, 1.0)
            business_non_viable = revenue < revenue_floor
            revenue = max(revenue, revenue_floor)
            t_idx = max(0,y-2025)
            margin_t   = max(margin_floor,ebitda_margin_0*(1-margin_erosion_rate*t_idx))
            ebitda     = revenue*margin_t - net_carbon_cost
            ebitda_adj = ebitda*(1-(1-G)*0.15)
            carbon_burden = net_carbon_cost/max(revenue,1)
            credit_spread = BASE_CREDIT_SPREAD*(1+spread_beta*carbon_burden)
            interest_stressed = interest_payment*(1+credit_spread)
            dscr = ebitda_adj/max(interest_stressed,1e-6)
            # FIX-04: signed gap; FIX-02: no gamma_phys; FIX-06: no delta_gdp direct
            dscr_gap = np.clip(1.5-dscr,-4.0,6.0)
            logit_pd = logit(base_pd)+alpha_dscr*dscr_gap+beta_carbon_credit*carbon_burden
            pd_t = np.clip(sigmoid(logit_pd),PD_FLOOR,PD_CAP)
            # Public credit-risk interpretation: climate adjustment should not reduce borrower risk below
            # the user-entered borrower base PD. Scenario-implied PD may be lower under benign cases, but
            # climate-adjusted PD is floored at base PD for consistency and market-facing credibility.
            pd_t = max(float(base_pd), float(pd_t))
            if business_non_viable:
                pd_t = PD_CAP
            lgd_t = np.clip(LGD_0*(1+0.2*carbon_burden+LGD_PHYSICAL_MULTIPLIER*physical_loss),0,1)
            ecl   = ecl_cr(pd_t, lgd_t, exposure_at_default)
            ratio = 0 if carbon_price_adj<P_START else min((carbon_price_adj-P_START)/(P_FULL-P_START),1)
            stranded = high_carbon_assets*min(ratio,MAX_STRANDING)
            req_capex = (scope1+scope2)*abatement_potential*abatement_cost/1e7
            capex_gap = req_capex-planned_capex
            results.append({"Scenario":scen,"Year":y,"Revenue":revenue,"EBITDA":ebitda_adj,
                "EBITDA_Margin":ebitda_adj/max(revenue,1),"Carbon_Burden":carbon_burden,
                "DSCR":dscr,"PD_Transition":pd_t,"LGD":lgd_t,"ECL_Transition":ecl,
                "Stranded_Assets":stranded,"CAPEX_Gap":capex_gap,
                "Physical_Loss":physical_loss,"GDP_Shock":gdp_shock,"Carbon_Price_Adj":carbon_price_adj,"Business_NonViable":business_non_viable})
    df = pd.DataFrame(results)
    if df.empty: raise ValueError("Transition engine produced no outputs.")
    return df.sort_values(["Scenario","Year"]).reset_index(drop=True)

# ============================================================
# PHYSICAL RISK HELPERS  (PHYS-01)
# ============================================================
def extract_ngfs_temperature_path(df_long, scenarios, reporting_year, years):
    """Return scenario temperature path using same-scenario reporting-year temperature as baseline.

    Previous versions compared each scenario against Current Policies in the same year, which made
    Current Policies show zero incremental physical risk over time. For physical risk projections,
    the clearer interpretation is absolute forward movement from the selected reporting-year baseline
    for each scenario.
    """
    records = []
    for scen in scenarios:
        d_s = df_long[df_long["Scenario"]==scen]
        temp_rows = d_s[d_s["Variable"].str.contains("Temperature",case=False,na=False)]
        if temp_rows.empty:
            continue
        # Use selected reporting year when available; otherwise nearest available year.
        avail_years = sorted(temp_rows["Year"].unique())
        base_year = reporting_year if reporting_year in avail_years else min(avail_years, key=lambda yy: abs(int(yy)-int(reporting_year)))
        base_val = temp_rows.loc[temp_rows["Year"]==base_year, "Value"]
        if base_val.empty:
            continue
        bv = float(base_val.iloc[0])
        for yr in years:
            tr = temp_rows.loc[temp_rows["Year"]==yr,"Value"]
            if tr.empty:
                continue
            tv = float(tr.iloc[0])
            records.append({
                "Scenario":scen,"Year":int(yr),"Temp_C":round(tv,3),
                "Baseline_Year":int(base_year),"Baseline_Temp_C":round(bv,3),
                "Delta_T_C":round(max(0.0,tv-bv),3),
                "Delta_T_vs_Baseline":round(max(0.0,tv-bv),3),
            })
    return pd.DataFrame(records)

def project_physical_risk_ngfs(df_ngfs_temp, baseline_damage_index, baseline_revenue_loss_cr,
        total_ebitda_cr, interest_cr, baseline_pd, lgd, ead_cr, gamma_phys=0.5,
        dscr_sensitivity=1.0, pd_floor=0.0005, pd_cap=0.35):
    """
    PHYS-01: Illustrative physical damage multiplier replaces flat 1+0.25*ΔT
    damage_mult = 1 + 0.20·ΔT + 0.04·ΔT²
    P10/P90 uncertainty: σ = 0.08·ΔT
    """
    records = []
    for _, row in df_ngfs_temp.iterrows():
        dt = row["Delta_T_C"] if "Delta_T_C" in row else row["Delta_T_vs_Baseline"]
        # Illustrative non-linear physical damage multiplier (replaces 0.25*dt)
        dm       = 1.0 + 0.20*dt + 0.04*dt**2
        sigma    = 0.08*dt
        dm_p10   = max(1.0, dm-1.645*sigma)
        dm_p90   = dm+1.645*sigma
        rl_p50   = baseline_revenue_loss_cr*dm
        rl_p10   = baseline_revenue_loss_cr*dm_p10
        rl_p90   = baseline_revenue_loss_cr*dm_p90
        # DSCR (physical stress path)
        ebitda_s = total_ebitda_cr - rl_p50*(total_ebitda_cr/max(baseline_revenue_loss_cr*10,1))
        dscr_p   = ebitda_s/max(interest_cr,1e-6)
        dscr_gap = np.clip(1.5-dscr_p,-4.0,6.0)  # FIX-04 signed
        logit_p  = logit(baseline_pd)+dscr_sensitivity*dscr_gap+gamma_phys*baseline_damage_index
        pd_p     = np.clip(sigmoid(logit_p),pd_floor,pd_cap)
        ecl_p    = ecl_cr(pd_p, lgd, ead_cr)
        chronic  = min(0.7,dt/3.0)
        records.append({
            "Scenario":row["Scenario"],"Year":int(row["Year"]),
            "Delta_T_C":round(dt,3),"Temp_C":round(row["Temp_C"],3),
            "Damage_Multiplier":round(dm,3),
            "Revenue_Loss_P10_Cr":round(rl_p10,2),
            "Revenue_Loss_P50_Cr":round(rl_p50,2),
            "Revenue_Loss_P90_Cr":round(rl_p90,2),
            "DSCR_Physical":round(dscr_p,3),"PD_Physical":round(pd_p,5),
            "ECL_Physical_Cr":round(ecl_p,3),
            "Chronic_Loss_Cr":round(rl_p50*chronic,2),
            "Acute_Loss_Cr":round(rl_p50*(1-chronic),2),
        })
    return pd.DataFrame(records).sort_values(["Scenario","Year"]).reset_index(drop=True)


# ============================================================
# ONE-CLICK PUBLIC DEMO RUNNER
# ============================================================
def _run_full_public_demo():
    """Populate all major demo outputs for the locked fictional company."""
    selected_scenarios = sorted(df_long["Scenario"].unique())
    df_transition = run_transition_engine(
        df_long, selected_scenarios, revenue_0, ebitda_margin_0,
        interest_payment, TOTAL_EMISSIONS, high_carbon_assets,
        exposure_at_default, carbon_pass_through, demand_elasticity,
        price_elasticity, beta_carbon_transition, beta_carbon_credit,
        USD_INR, G, base_pd, LGD_0, abatement_cost, abatement_potential,
        planned_capex, P_START, P_FULL, MAX_STRANDING, margin_erosion_rate,
        margin_floor, alpha_dscr, tax_rate, CARBON_PRICE_INFLATION_FACTOR,
        delta_gdp=delta_gdp
    )
    st.session_state["df_transition"] = df_transition
    st.session_state["transition_ran"] = True
    st.session_state["df_transition_summary"] = df_transition.groupby("Scenario").agg({
        "Carbon_Burden":"max","EBITDA_Margin":"min","DSCR":"min",
        "PD_Transition":"max","ECL_Transition":"max",
        "Stranded_Assets":"max","CAPEX_Gap":"max",
    }).round(4).reset_index()

    # Demo physical risk: locked assets, proportional EAD, p90 flood extraction.
    df = pd.DataFrame({
        "asset_id":["A1","A2","A3"],
        "asset_type":["Steel Plant","Rolling Mill","Port Logistics Yard"],
        "latitude":[22.80,23.55,20.32],"longitude":[86.20,87.32,86.61],
        "revenue_inr_cr":[3500.0,2000.0,1200.0],
        "base_pd":[base_pd,base_pd,base_pd],"lgd":[LGD_0,LGD_0,LGD_0],
    })
    V={"flood":0.6,"heat":0.7,"cyclone":0.6}; KAPPA={"flood":0.4,"heat":0.2,"cyclone":0.5}
    MAX_DOWNTIME={"Steel Plant":90,"Rolling Mill":60,"Port Logistics Yard":45}
    _ensure_gis_files()
    flood_vals=[]
    try:
        if rasterio is None or rowcol is None or Window is None:
            raise RuntimeError(RASTERIO_IMPORT_ERROR or "rasterio unavailable")
        with rasterio.open(r"Data/floodMapGL_rp100y.tif") as src:
            pk=max(abs(src.res[0])*111, 1e-6); bp=max(1, int(1/pk))
            for _,a in df.iterrows():
                try:
                    ri,ci=rowcol(src.transform,a["longitude"],a["latitude"])
                    r0=max(0,ri-bp); r1=min(src.height,ri+bp); c0=max(0,ci-bp); c1=min(src.width,ci+bp)
                    if r1<=r0 or c1<=c0:
                        flood_vals.append(0.0); continue
                    w=src.read(1, window=Window(c0,r0,c1-c0,r1-r0), masked=True)
                    v=w.compressed() if hasattr(w, "compressed") else w.ravel()
                    v=v[(v>0)&(v<100)]
                    flood_vals.append(float(np.percentile(v,90)) if v.size else 0.0)
                except Exception:
                    flood_vals.append(0.0)
    except Exception:
        flood_vals=[0.0]*len(df)
    df["flood_depth_m"]=flood_vals
    try:
        heat=pd.read_csv(r"Data/era5_test_day_grid.csv")
        heat["exceed"]=(heat["temp_c"]>=35).astype(int)
        hg=heat.groupby(["lat","lon"],as_index=False)["exceed"].sum()
    except Exception:
        hg=pd.DataFrame(columns=["lat","lon","exceed"])
    def nearest_heat(row):
        if hg.empty: return 0
        idx=((hg["lat"]-row["latitude"])**2+(hg["lon"]-row["longitude"])**2).idxmin()
        return int(hg.loc[idx,"exceed"])
    df["heat_days"]=df.apply(nearest_heat,axis=1)
    try:
        cyc=pd.read_csv(r"Data/ibtracs.NI.list.v04r01.csv", usecols=["LAT","LON","USA_WIND"], low_memory=False)
        cyc=cyc.apply(pd.to_numeric,errors="coerce").dropna(); cyc["wind_kmh"]=cyc["USA_WIND"]*1.852
    except Exception:
        cyc=pd.DataFrame(columns=["LAT","LON","wind_kmh"])
    def max_wind(a):
        if cyc.empty: return 0.0
        try:
            lat1=np.radians(float(a["latitude"])); lon1=np.radians(float(a["longitude"]))
            lat2=np.radians(cyc["LAT"].astype(float).to_numpy()); lon2=np.radians(cyc["LON"].astype(float).to_numpy())
            dlat=lat2-lat1; dlon=lon2-lon1
            h=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            d=6371*2*np.arcsin(np.sqrt(h)); vals=cyc.loc[d<=100,"wind_kmh"]
            return float(vals.max()) if len(vals)>0 else 0.0
        except Exception:
            return 0.0
    df["max_wind_kmh"]=df.apply(max_wind,axis=1)
    df["H_flood"]=df["flood_depth_m"].apply(flood_damage)
    df["H_heat"]=df["heat_days"].apply(heat_damage)
    df["H_cyclone"]=df["max_wind_kmh"].apply(cyclone_damage)
    df["D_total"]=(df["H_flood"]*V["flood"]*KAPPA["flood"]+df["H_heat"]*V["heat"]*KAPPA["heat"]+df["H_cyclone"]*V["cyclone"]*KAPPA["cyclone"]).clip(0,1)
    df["downtime_days"]=df.apply(lambda r: r["D_total"]*MAX_DOWNTIME.get(r["asset_type"],60),axis=1)
    df["revenue_loss"]=df["downtime_days"]/365*df["revenue_inr_cr"]
    total_rev_loss=df["revenue_loss"].sum(); delta_ebitda=total_rev_loss*ebitda_margin_0
    dscr_phys=(df["revenue_inr_cr"].sum()*ebitda_margin_0-delta_ebitda)/max(interest_payment,1e-6)
    df["PD_Physical"]=(df["base_pd"]*(1+gamma_phys*df["D_total"])).clip(0,1)
    revenue_weight = df["revenue_inr_cr"] / max(df["revenue_inr_cr"].sum(), 1e-6)
    coverage = float(df["revenue_inr_cr"].sum()/max(revenue_0,1e-6))
    effective_ead = exposure_at_default * min(max(coverage,0),1)
    df["EAD_allocated_cr"] = effective_ead * revenue_weight
    df["Baseline_ECL_asset"] = df["base_pd"]*df["lgd"]*df["EAD_allocated_cr"]
    df["ECL_asset"] = df["PD_Physical"]*df["lgd"]*df["EAD_allocated_cr"]
    df["Delta_ECL_asset"] = (df["ECL_asset"]-df["Baseline_ECL_asset"]).clip(lower=0)
    pd_phys=float((df["PD_Physical"]*revenue_weight).sum())
    physical_ecl=float(df["ECL_asset"].sum()); delta_ecl=float(df["Delta_ECL_asset"].sum())
    df_ngfs_temp = extract_ngfs_temperature_path(df_long, selected_scenarios, REPORTING_YEAR, sorted(df_transition["Year"].unique()))
    df_phys_proj = project_physical_risk_ngfs(df_ngfs_temp, df["D_total"].mean(), total_rev_loss, df["revenue_inr_cr"].sum()*ebitda_margin_0, interest_payment, pd_phys, LGD_0, effective_ead, gamma_phys)
    st.session_state["phys_assets"]=df
    st.session_state["phys_summary"]={"Total Revenue Loss (₹ Cr)":total_rev_loss,"EBITDA Loss (₹ Cr)":delta_ebitda,"Post-Risk DSCR":dscr_phys,"Physical Risk PD":pd_phys,"Physical ECL (₹ Cr)":physical_ecl,"ΔECL (₹ Cr)":delta_ecl,"Assets Analysed":len(df),"Asset Revenue Coverage %":coverage*100,"Effective Physical EAD (₹ Cr)":effective_ead,"Physical EAD Allocation Mode":"Company-wide proportional exposure", "Reporting Year": REPORTING_YEAR}
    st.session_state["df_physical_projection"]=df_phys_proj
    st.session_state["physical_ran"]=True

    # Locked BRSR demo diagnostics.
    ghg_bench=SECTOR_GHG_BENCHMARKS.get(sector,SECTOR_GHG_BENCHMARKS["Manufacturing"])
    en_bench=SECTOR_ENERGY_BENCHMARKS.get(sector,SECTOR_ENERGY_BENCHMARKS["Manufacturing"])
    total_energy_kwh=1_000_000_000; renewable_share_pct=15; total_water_m3=50_000_000; water_stress_region="High"; recycled_water_pct=10; total_waste_mt=100_000; hazardous_waste_pct=20; target_coverage_pct=50
    has_verified_data=False; has_scope3=True; has_board_oversight=False; has_net_zero_target=False
    ghg_int=TOTAL_EMISSIONS/max(revenue_0,1e-6); s1_int=scope1/max(revenue_0,1e-6); s2_int=scope2/max(revenue_0,1e-6); s3_int=scope3/max(revenue_0,1e-6)
    en_int=total_energy_kwh/max(revenue_0*1e6,1); wa_int=total_water_m3/max(revenue_0*1e6,1)
    flags=[]
    if ghg_int>ghg_bench["p75"]: flags.append("High GHG intensity (>P75)")
    if renewable_share_pct<20: flags.append("Low renewable energy (<20%)")
    if en_int>en_bench["p75"]: flags.append("High energy intensity (>P75)")
    if water_stress_region=="High": flags.append("High water stress + high withdrawal")
    if target_coverage_pct<50: flags.append("Low target coverage (<50%)")
    if hazardous_waste_pct>30: flags.append("High hazardous waste (>30%)")
    if not has_verified_data: flags.append("No verified emissions data")
    if not has_scope3: flags.append("Missing Scope 3 disclosure")
    pd_adj=float(np.clip(sum(BRSR_PD_SPREAD.get(f,0) for f in flags),0,0.015))
    esc_r={"Low":0.03,"Medium":0.08,"High":0.18}[water_stress_region]
    w_cum_cost=sum(total_water_m3*42*((1+esc_r)**yr-(1+esc_r)**(yr-1))/1e7 for yr in range(1,6))
    fossil_kwh=total_energy_kwh*(100-renewable_share_pct)/100; energy_tr_risk=fossil_kwh*2.0/1e7
    rdness_items={"Renewable ≥20%":1.0 if renewable_share_pct>=20 else 0.5,"Target Coverage ≥50%":1.0,"Hazardous Waste ≤30%":1.0,"Water Risk Mgmt":0.0,"Scope 3 Disclosed":1.0,"Verified Data":0.0,"Board Oversight":0.0,"Long-term Climate Target":0.0}
    readiness_score=sum(rdness_items.values())/len(rdness_items)*100
    risk_score_brsr=12+18+7+8+6
    brsr_summary=pd.DataFrame([{"GHG_Intensity":round(ghg_int,3),"S1_Intensity":round(s1_int,3),"S2_Intensity":round(s2_int,3),"S3_Intensity":round(s3_int,3),"Energy_Intensity":round(en_int,3),"Renewable_%":renewable_share_pct,"Water_Intensity":round(wa_int,3),"Recycled_Water_%":recycled_water_pct,"Hazardous_Waste_%":hazardous_waste_pct,"Target_Coverage_%":target_coverage_pct,"BRSR_PD_Adj_bps":round(pd_adj*10000,1),"Water_Cost_Risk_Cr":round(w_cum_cost,2),"Energy_TR_Risk_Cr":round(energy_tr_risk,2),"Readiness_Score":round(readiness_score,1),"Overall_Risk_Score":risk_score_brsr}])
    st.session_state["brsr_summary"]=brsr_summary
    st.session_state["brsr_flags"]=pd.DataFrame({"Flag":flags})
    st.session_state["brsr_pd_adj"]=pd_adj
    st.session_state["brsr_ran"]=True

    # Integrated board summary.
    pd_trans=float(df_transition["PD_Transition"].max()); pd_phys_scen=max(pd_phys, float(df_phys_proj["PD_Physical"].max()) if isinstance(df_phys_proj,pd.DataFrame) and not df_phys_proj.empty else pd_phys)
    pd_copula, _joint = combined_pd_union(pd_trans, pd_phys_scen, default_correlation)
    pd_integrated=float(np.clip(pd_copula*brsr_governance_multiplier(pd_adj),PD_FLOOR,PD_CAP))
    lgd_integrated=float(np.clip(max(LGD_0,df_transition["LGD"].max()),0,1))
    ecl_integrated=ecl_cr(pd_integrated, lgd_integrated, exposure_at_default)
    dscr_integrated=float(min(df_transition["DSCR"].min(), dscr_phys))
    ecl_ead_ratio=ecl_integrated/exposure_at_default if exposure_at_default>0 else 0
    pd_score=min(100,(pd_integrated/0.20)*100); ecl_score=min(100,(ecl_ead_ratio/0.05)*100)
    dscr_score=100 if dscr_integrated<1.0 else 60 if dscr_integrated<1.2 else 30 if dscr_integrated<1.5 else 10
    risk_score_int=round(0.4*pd_score+0.4*ecl_score+0.2*dscr_score,1)
    capital_signal="Severe" if ecl_ead_ratio>=0.05 else "Elevated" if ecl_ead_ratio>=0.03 else "Moderate" if ecl_ead_ratio>=0.015 else "Limited"
    st.session_state["df_integrated_summary"]=pd.DataFrame([
        {"Metric":"Integrated PD (Union PD × BRSR factor)","Value":f"{pd_integrated:.4f}","ISSB S2":"§15(a)"},
        {"Metric":"Integrated ECL (₹ Cr)","Value":f"{ecl_integrated:.2f}","ISSB S2":"§15(b)"},
        {"Metric":"Post-Stress DSCR","Value":f"{dscr_integrated:.2f}","ISSB S2":"§15(c)"},
        {"Metric":"ECL / EAD","Value":f"{ecl_ead_ratio:.4f}","ISSB S2":"§16"},
        {"Metric":"BRSR Governance Factor","Value":f"x{brsr_governance_multiplier(pd_adj):.3f}","ISSB S2":"§15(a) operational"},
        {"Metric":"Capital Stress Signal","Value":capital_signal,"ISSB S2":"§16"},
        {"Metric":"Integrated Risk Score (0–100)","Value":f"{risk_score_int}","ISSB S2":"§14–16"},
        {"Metric":"Stranded Assets (₹ Cr)","Value":f"{df_transition['Stranded_Assets'].max():.0f}","ISSB S2":"§22"},
        {"Metric":"CAPEX Gap / Surplus (₹ Cr)","Value":_fmt_capex_position(df_transition['CAPEX_Gap'].max()),"ISSB S2":"§14"},
    ])
    st.session_state["integrated_ran"]=True
    st.session_state["run_full_demo_requested"]=False


# One-click full demo auto-run removed for public demo. Modules are run from their own tabs and cached until reset.

# ============================================================
# TAB LAYOUT
# ============================================================
# Tab order:  Dashboard → Transition → Physical → BRSR Core → Targets → Integrated → Plots → AI → …
# BRSR is placed BEFORE Targets so BRSR results feed into BRSR-based target planning in Targets tab.
(dashboard_tab, transition_tab, physical_tab, brsr_tab, targets_tab,
 integrated_tab, plots_tab, ai_tab, methodology_tab, validation_tab, calibration_tab,
 access_tab,
) = st.tabs([
    "🏠 Dashboard",
    "⚡ Transition Risk",
    "🌍 Physical Risk",
    "📘 BRSR Core",
    "🎯 Targets",
    "🧩 Integrated Risk",
    "📈 Plots",
    "🤖 AI Narrative",
    "🛡️ Model Governance",
    "🧪 Pilot Readiness",
    "🔒 Enterprise Preview",
    "🚀 Get Access",
])

# ============================================================
# TAB 0 — DASHBOARD  (UI-03)
# ============================================================
with dashboard_tab:
    st.markdown(f"<h2 style='color:{C['white']};margin-bottom:4px;'>Climate Risk Dashboard</h2>", unsafe_allow_html=True)
    st.caption(f"**{company_name}** · {sector} · Reporting Year {REPORTING_YEAR}")

    # Module status badges
    badges = []
    for label, key, color in [
        ("⚡ Transition","transition_ran",C["accent2"]),
        ("🌍 Physical","physical_ran",C["coral"]),
        ("🎯 Targets","targets_ran",C["amber"]),
        ("📘 BRSR","brsr_ran",C["mint"]),
    ]:
        ran = st.session_state.get(key,False)
        bg  = color if ran else C["bg_mid"]
        tc  = C["bg_dark"] if ran else C["slate"]
        badges.append(f"<span style='background:{bg};color:{tc};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;'>{label} {'✓' if ran else '—'}</span>")
    st.markdown("<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;'>"+"".join(badges)+"</div>",unsafe_allow_html=True)

    transition_ran = st.session_state.get("transition_ran",False)
    physical_ran   = st.session_state.get("physical_ran",False)
    brsr_ran       = st.session_state.get("brsr_ran",False)
    integrated_ran = st.session_state.get("integrated_ran",False)

    if not (transition_ran or physical_ran or brsr_ran):
        # ── Welcome / Demo CTA ──
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{C['card']},{C['bg_mid']});
                    border:1px solid {C['bg_ocean']};border-radius:14px;
                    padding:36px 40px;text-align:center;margin-top:8px;">
          <div style="font-size:40px;margin-bottom:10px;">🌍</div>
          <div style="font-size:22px;font-weight:700;color:{C['white']};margin-bottom:6px;">
            Welcome to ICCRE
          </div>
          <div style="font-size:14px;color:{C['accent2']};margin-bottom:4px;font-weight:600;">
            {PRODUCT_TAGLINE}
          </div>
          <div style="font-size:12px;color:{C['slate']};margin-bottom:24px;">
            {PRODUCT_SUBTITLE}
          </div>
          <div style="display:flex;justify-content:center;gap:32px;flex-wrap:wrap;margin-bottom:24px;">
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:{C['accent2']};">3</div>
              <div style="font-size:10px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;">NGFS Scenarios</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:{C['mint']};">5</div>
              <div style="font-size:10px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;">Indian Sectors</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:{C['amber']};">11</div>
              <div style="font-size:10px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;">Analytics Modules</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:28px;font-weight:700;color:{C['coral']};">~20</div>
              <div style="font-size:10px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;">Minutes to Full Report</div>
            </div>
          </div>
          <div style="background:{C['bg_dark']};border-radius:10px;padding:16px 20px;
                      margin-bottom:20px;text-align:left;max-width:520px;margin-left:auto;margin-right:auto;">
            <div style="font-size:11px;font-weight:700;color:{C['amber']};margin-bottom:8px;
                        text-transform:uppercase;letter-spacing:.06em;">⚡ Fastest Start — 3 Steps</div>
            <div style="font-size:12px;color:{C['off_white']};line-height:2.0;">
              1️⃣ Select a <strong style="color:{C['amber']};">Reporting Year</strong> in the sidebar<br>
              2️⃣ Go to <strong style="color:{C['accent2']};">⚡ Transition Risk</strong> tab → Click Run<br>
              3️⃣ Compare reporting-year snapshot vs multi-year scenario pathway
            </div>
          </div>
          <div style="display:flex;justify-content:center;gap:12px;flex-wrap:wrap;">
            <div style="background:{C['bg_dark']};border:1px solid {C['bg_mid']};border-radius:8px;
                        padding:10px 16px;text-align:center;min-width:150px;">
              <div style="font-size:10px;font-weight:700;color:{C['mint']};text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:4px;">Aligned with</div>
              <div style="font-size:11px;color:{C['off_white']};">ISSB IFRS S2</div>
              <div style="font-size:11px;color:{C['off_white']};">RBI 2024 Guidelines</div>
              <div style="font-size:11px;color:{C['off_white']};">SEBI BRSR Core</div>
            </div>
            <div style="background:{C['bg_dark']};border:1px solid {C['bg_mid']};border-radius:8px;
                        padding:10px 16px;text-align:center;min-width:150px;">
              <div style="font-size:10px;font-weight:700;color:{C['amber']};text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:4px;">Outputs include</div>
              <div style="font-size:11px;color:{C['off_white']};">PD · LGD · ECL</div>
              <div style="font-size:11px;color:{C['off_white']};">DSCR · Climate VaR</div>
              <div style="font-size:11px;color:{C['off_white']};">ICAAP Capital Signal</div>
            </div>
            <div style="background:{C['bg_dark']};border:1px solid {C['bg_mid']};border-radius:8px;
                        padding:10px 16px;text-align:center;min-width:150px;">
              <div style="font-size:10px;font-weight:700;color:{C['coral']};text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:4px;">Zero competitors</div>
              <div style="font-size:11px;color:{C['off_white']};">India-native</div>
              <div style="font-size:11px;color:{C['off_white']};">INR-denominated</div>
              <div style="font-size:11px;color:{C['off_white']};">BRSR-native</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Pull live metrics
        pd_t = ecl_t = dscr_t = stranded_t = capex_t = pd_p = ecl_p = brsr_pd = risk_score = None
        pd_t_note = ecl_t_note = dscr_t_note = stranded_t_note = ""
        if transition_ran:
            df_t = st.session_state.get("df_transition")
            if isinstance(df_t,pd.DataFrame) and not df_t.empty:
                pd_t, _pd_y_dash, _pd_s_dash = _metric_extreme(df_t, "PD_Transition", "max")
                ecl_t, _ecl_y_dash, _ecl_s_dash = _metric_extreme(df_t, "ECL_Transition", "max")
                dscr_t, _dscr_y_dash, _dscr_s_dash = _metric_extreme(df_t, "DSCR", "min")
                stranded_t, _str_y_dash, _str_s_dash = _metric_extreme(df_t, "Stranded_Assets", "max")
                capex_t = df_t["CAPEX_Gap"].max()
                pd_t_note = _year_scenario_note(_pd_y_dash, _pd_s_dash)
                ecl_t_note = _year_scenario_note(_ecl_y_dash, _ecl_s_dash)
                dscr_t_note = _year_scenario_note(_dscr_y_dash, _dscr_s_dash, prefix="Worst")
                stranded_t_note = _year_scenario_note(_str_y_dash, _str_s_dash)
        if physical_ran:
            ps = st.session_state.get("phys_summary",{})
            if isinstance(ps,dict):
                pd_p  = ps.get("Physical Risk PD",0)
                ecl_p = ps.get("ΔECL (₹ Cr)",0)
        if brsr_ran:
            bs = st.session_state.get("brsr_summary")
            if isinstance(bs,pd.DataFrame) and not bs.empty:
                brsr_pd   = st.session_state.get("brsr_pd_adj",0)
                risk_score = bs.iloc[0].get("Overall_Risk_Score",None)

        # Reporting-year snapshot block for transition results
        if transition_ran and isinstance(st.session_state.get("df_transition"), pd.DataFrame) and not st.session_state["df_transition"].empty:
            _df_ry_dash, _actual_ry_dash = reporting_year_slice(st.session_state["df_transition"], REPORTING_YEAR)
            if isinstance(_df_ry_dash, pd.DataFrame) and not _df_ry_dash.empty:
                _ry_worst_dash = _df_ry_dash.loc[_df_ry_dash["PD_Transition"].idxmax()]
                scope_badge("single", f"Dashboard reporting-year snapshot: selected {REPORTING_YEAR}; displayed year {_actual_ry_dash}.")
                render_metric_grid([
                    {"title":f"{_actual_ry_dash} Worst-Scenario PD", "value":_fmt_pct(_ry_worst_dash['PD_Transition']), "subtitle":f"Scenario: {_ry_worst_dash['Scenario']}", "accent":C["accent2"], "scope":"Reporting year"},
                    {"title":f"{_actual_ry_dash} Worst-Scenario ECL", "value":_fmt_money_cr(_ry_worst_dash['ECL_Transition']), "subtitle":"Reporting-year credit-loss view", "accent":C["amber"], "scope":"Reporting year"},
                    {"title":f"{_actual_ry_dash} DSCR", "value":_fmt_num(_ry_worst_dash['DSCR'], "×", 2), "subtitle":"Reporting-year debt-service capacity", "accent":C["coral"] if _ry_worst_dash['DSCR']<1.2 else C["mint"], "scope":"Reporting year"},
                ], columns=3)
                scope_badge("multi", "Dashboard peak cards below show multi-year scenario peak/worst values.")

        # KPI row
        kpis = []
        if pd_t is not None:   kpis.append(("Peak Transition PD",f"{pd_t:.2%}",C["accent2"],pd_t_note or "Multi-year scenario peak"))
        if ecl_t is not None:  kpis.append(("Peak ECL",f"₹{ecl_t:,.1f} Cr",C["amber"],ecl_t_note or "Multi-year scenario peak"))
        if dscr_t is not None: kpis.append(("Worst DSCR",f"{dscr_t:.2f}×",C["coral"] if dscr_t<1.2 else C["mint"],dscr_t_note or "Multi-year scenario worst"))
        if pd_p is not None:   kpis.append(("Physical PD",f"{pd_p:.2%}",C["coral"],"Asset-level physical risk"))
        if brsr_pd is not None:kpis.append(("BRSR Governance Signal",f"+{brsr_pd*10000:.0f} bps",C["purple"],"Operational climate risk"))
        if stranded_t is not None: kpis.append(("Stranded Assets",f"₹{stranded_t:,.0f} Cr",C["amber"],stranded_t_note or "High-carbon asset risk"))

        cols = st.columns(min(len(kpis),6))
        for col_obj,(label,val,color,hint) in zip(cols,kpis):
            with col_obj:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-value" style="color:{color};">{val}</div>
                  <div class="kpi-label">{label}</div>
                  <div style="font-size:10px;color:{C['slate']};margin-top:4px;">{hint}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px;'></div>",unsafe_allow_html=True)

        # PD sparkline + risk gauge side by side
        if transition_ran and isinstance(st.session_state.get("df_transition"),pd.DataFrame):
            df_t = st.session_state["df_transition"]
            col_spark, col_gauge = st.columns([3,1])

            with col_spark:
                fig_spark = go.Figure()
                for scen in df_t["Scenario"].unique():
                    ds = df_t[df_t["Scenario"]==scen]
                    fig_spark.add_trace(go.Scatter(
                        x=ds["Year"],y=ds["PD_Transition"],mode="lines+markers",
                        name=scen[:28],line=dict(color=_scen_color(scen),width=2.5),
                        marker=dict(size=6),
                        hovertemplate="<b>%{x}</b><br>PD: %{y:.2%}<extra>"+scen[:20]+"</extra>"
                    ))
                fig_spark.update_layout(**_chart_layout("Transition PD Trajectory — All Scenarios",260))
                fig_spark.update_yaxes(tickformat=".1%")
                _ax_style(fig_spark)
                st.plotly_chart(fig_spark,width="stretch")

            with col_gauge:
                if pd_t is not None:
                    gauge_color = C["coral"] if pd_t>0.10 else C["amber"] if pd_t>0.05 else C["mint"]
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=pd_t*100,
                        number={"suffix":"%","font":{"color":gauge_color,"size":22}},
                        gauge={
                            "axis":{"range":[0,30],"tickcolor":C["slate"]},
                            "bar":{"color":gauge_color},
                            "bgcolor":C["bg_dark"],
                            "steps":[
                                {"range":[0,5],"color":C["bg_mid"]},
                                {"range":[5,10],"color":"#1A3A2A"},
                                {"range":[10,20],"color":"#3A1A1A"},
                                {"range":[20,30],"color":"#4A1A1A"},
                            ],
                            "threshold":{"line":{"color":"white","width":2},"thickness":0.75,"value":pd_t*100}
                        },
                        title={"text":"Peak PD","font":{"color":C["slate"],"size":12}}
                    ))
                    fig_g.update_layout(height=200,paper_bgcolor=C["bg_dark"],margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig_g,width="stretch")

        # BRSR + Physical summary row
        if brsr_ran or physical_ran:
            bcol, pcol = st.columns(2)
            if brsr_ran and risk_score is not None:
                with bcol:
                    sev_color = C["mint"] if risk_score<30 else C["amber"] if risk_score<60 else C["coral"]
                    sev_text  = "Low" if risk_score<30 else "Moderate" if risk_score<60 else "High"
                    st.markdown(f"""
                    <div style="background:{C['card']};border:1px solid {C['bg_ocean']};border-radius:10px;padding:14px 16px;">
                      <div style="font-size:12px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">BRSR Operational Risk</div>
                      <div style="font-size:26px;font-weight:600;color:{sev_color};font-family:monospace;">{risk_score}/100</div>
                      <div style="font-size:13px;color:{sev_color};font-weight:600;">{sev_text} Risk · +{brsr_pd*10000:.0f} bps risk overlay</div>
                    </div>""",unsafe_allow_html=True)
            if physical_ran and pd_p is not None:
                with pcol:
                    pc = C["coral"] if pd_p>0.05 else C["amber"] if pd_p>0.02 else C["mint"]
                    st.markdown(f"""
                    <div style="background:{C['card']};border:1px solid {C['bg_ocean']};border-radius:10px;padding:14px 16px;">
                      <div style="font-size:12px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Physical Risk (Portfolio)</div>
                      <div style="font-size:26px;font-weight:600;color:{pc};font-family:monospace;">{pd_p:.2%}</div>
                      <div style="font-size:13px;color:{C['slate']};">ΔECL: ₹{ecl_p:.2f} Cr · Asset-level aggregation</div>
                    </div>""",unsafe_allow_html=True)

        # Integrated risk summary — shown once integrated tab has been computed
        if integrated_ran:
            df_int_dash = st.session_state.get("df_integrated_summary")
            st.markdown(f"<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:13px;color:{C['slate']};text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;'>🧩 Integrated Risk (latest run)</div>", unsafe_allow_html=True)
            # Pull integrated metrics from the stored summary table
            _pd_int = _ecl_int = _risk_sc = _capital = None
            if isinstance(df_int_dash, pd.DataFrame) and not df_int_dash.empty:
                def _get(metric_substr):
                    rows = df_int_dash[df_int_dash["Metric"].str.contains(metric_substr, case=False, na=False)]
                    return rows["Value"].iloc[0] if not rows.empty else None
                _pd_int   = _get("Integrated PD")
                _ecl_int  = _get("Integrated ECL")
                _capital  = _get("Capital Stress")
                _risk_sc  = _get("Risk Score")

            if _risk_sc is not None:
                try:
                    _rs_val = float(str(_risk_sc))
                    _rag_c  = C["mint"] if _rs_val < 30 else C["amber"] if _rs_val < 60 else C["coral"] if _rs_val < 80 else "#F72585"
                    _rag_t  = "Low" if _rs_val < 30 else "Moderate" if _rs_val < 60 else "Severe" if _rs_val < 80 else "Extreme"
                except Exception:
                    _rag_c = C["amber"]; _rag_t = "—"; _rs_val = 0

                ic1, ic2, ic3, ic4 = st.columns(4)
                ic1.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:{_rag_c};">{_rag_t}</div><div class="kpi-label">Climate Risk Level</div></div>""", unsafe_allow_html=True)
                ic2.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:{C['accent2']};">{_pd_int or '—'}</div><div class="kpi-label">Integrated PD</div></div>""", unsafe_allow_html=True)
                ic3.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:{C['amber']};">{_ecl_int or '—'} Cr</div><div class="kpi-label">Integrated ECL</div></div>""", unsafe_allow_html=True)
                ic4.markdown(f"""<div class="kpi-card"><div class="kpi-value" style="color:{C['coral'] if _capital in ['Severe','Elevated'] else C['mint']};">{_capital or '—'}</div><div class="kpi-label">Capital Signal</div></div>""", unsafe_allow_html=True)

# ============================================================
# TAB 1 — TRANSITION RISK
# ============================================================
with transition_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>⚡ Transition Risk Engine</h2>",unsafe_allow_html=True)
    st.caption("NGFS scenario-driven financial transmission: Carbon price → Revenue → DSCR → PD → ECL")

    if not st.session_state.get("enable_transition",False):
        st.info("Enable **Transition Risk** in the sidebar to proceed.")
    else:
        # Public mode: internal correction notes hidden.

        all_scenarios = PUBLIC_DEMO_SCENARIOS if PUBLIC_DEMO_SCENARIOS else sorted(df_long["Scenario"].unique())
        selected_scenarios = st.multiselect(
            "Public Demo Scenarios",
            options=all_scenarios,
            default=all_scenarios,
            help="Public demo is limited to Current Policies, NDCs and Below 2°C scenarios."
        )
        st.caption("Scenario outputs use only: Current Policies, NDCs and Below 2°C. No other NGFS scenarios are shown in this demo.")
        run_btn = st.button("▶ Run Transition Risk Engine",type="primary")

        if not run_btn:
            if not render_cached_transition_results():
                st.info("Select scenarios and click **Run Transition Risk Engine**.")
        elif not selected_scenarios:
            st.warning("Select at least one scenario.")
        else:
            with st.spinner("Running transition risk engine..."):
                df_transition = run_transition_engine(
                    df_long,selected_scenarios,revenue_0,ebitda_margin_0,
                    interest_payment,TOTAL_EMISSIONS,high_carbon_assets,
                    exposure_at_default,carbon_pass_through,demand_elasticity,
                    price_elasticity,beta_carbon_transition,beta_carbon_credit,
                    USD_INR,G,base_pd,LGD_0,abatement_cost,abatement_potential,
                    planned_capex,P_START,P_FULL,MAX_STRANDING,margin_erosion_rate,
                    margin_floor,alpha_dscr,tax_rate,CARBON_PRICE_INFLATION_FACTOR,
                    delta_gdp=delta_gdp
                )
            st.session_state["df_transition"]  = df_transition
            st.session_state["transition_ran"] = True

            scope_badge("multi", "Transition outputs are calculated across all selected scenario years; headline cards show peak/worst value across the scenario path.")
            # Scenario summary table
            df_sum = (df_transition.groupby("Scenario").agg({
                "Carbon_Burden":"max","EBITDA_Margin":"min","DSCR":"min",
                "PD_Transition":"max","ECL_Transition":"max",
                "Stranded_Assets":"max","CAPEX_Gap":"max",
            }).round(4).reset_index())

            # Styled summary
            st.markdown(f"<h3 style='color:{C['white']};margin-top:16px;'>Scenario Risk Summary</h3>",unsafe_allow_html=True)

            def _color_pd(val):
                if isinstance(val,float):
                    if val>0.10: return f"color:{C['coral']};font-weight:600"
                    if val>0.05: return f"color:{C['amber']};font-weight:600"
                    return f"color:{C['mint']};font-weight:600"
                return ""

            st.dataframe(
                df_sum.style
                    .format({"Carbon_Burden":"{:.2%}","EBITDA_Margin":"{:.2%}","DSCR":"{:.2f}",
                              "PD_Transition":"{:.3%}","ECL_Transition":"{:.1f}",
                              "Stranded_Assets":"{:.0f}","CAPEX_Gap":"{:.0f}"})
                    .map(_color_pd, subset=["PD_Transition"])
                    .background_gradient(subset=["ECL_Transition"],cmap="Reds")
                    .set_properties(**{"background-color":C["bg_dark"],"color":C["text"]}),
                width="stretch", hide_index=True
            )

            # Headline KPIs — multi-year scenario peaks/worst values
            st.markdown("<h3 style='margin-top:16px;'>Headline Indicators</h3>",unsafe_allow_html=True)
            _pd_v, _pd_y, _pd_s = _metric_extreme(df_transition, "PD_Transition", "max")
            _ecl_v, _ecl_y, _ecl_s = _metric_extreme(df_transition, "ECL_Transition", "max")
            _dscr_v, _dscr_y, _dscr_s = _metric_extreme(df_transition, "DSCR", "min")
            _str_v, _str_y, _str_s = _metric_extreme(df_transition, "Stranded_Assets", "max")
            render_metric_grid([
                {"title":"Peak Transition PD", "value":_fmt_pct(_pd_v), "subtitle":_year_scenario_note(_pd_y, _pd_s), "accent":C["accent2"], "scope":"Multi-year scenario"},
                {"title":"Peak ECL", "value":_fmt_money_cr(_ecl_v), "subtitle":_year_scenario_note(_ecl_y, _ecl_s), "accent":C["amber"], "scope":"Multi-year scenario"},
                {"title":"Minimum DSCR", "value":_fmt_num(_dscr_v, "×", 2), "subtitle":_year_scenario_note(_dscr_y, _dscr_s, prefix="Worst"), "accent":C["coral"] if (_dscr_v is not None and _dscr_v<1.2) else C["mint"], "scope":"Multi-year scenario"},
                {"title":"Peak Stranded Assets", "value":_fmt_money_cr(_str_v, 0), "subtitle":_year_scenario_note(_str_y, _str_s), "accent":C["purple"], "scope":"Multi-year scenario"},
            ], columns=4)

            # Charts
            fig_pd = go.Figure()
            for scen in df_transition["Scenario"].unique():
                ds = df_transition[df_transition["Scenario"]==scen]
                fig_pd.add_trace(go.Scatter(x=ds["Year"],y=ds["PD_Transition"],mode="lines+markers",
                    name=scen[:30],line=dict(color=_scen_color(scen),width=2.5),marker=dict(size=6)))
            fig_pd.update_layout(**_chart_layout("PD Trajectory by NGFS Scenario",320))
            fig_pd.update_yaxes(tickformat=".1%")
            _ax_style(fig_pd)
            st.plotly_chart(fig_pd,width="stretch")

            fig_ecl = go.Figure()
            for scen in df_transition["Scenario"].unique():
                ds = df_transition[df_transition["Scenario"]==scen]
                fig_ecl.add_trace(go.Bar(x=ds["Year"],y=ds["ECL_Transition"],name=scen[:30],
                    marker_color=_scen_color(scen),opacity=0.85))
            fig_ecl.update_layout(**_chart_layout("Expected Credit Loss (₹ Cr) — All Scenarios",300),barmode="group")
            _ax_style(fig_ecl)
            st.plotly_chart(fig_ecl,width="stretch")

            fig_dscr = go.Figure()
            for scen in df_transition["Scenario"].unique():
                ds = df_transition[df_transition["Scenario"]==scen]
                fig_dscr.add_trace(go.Scatter(x=ds["Year"],y=ds["DSCR"],mode="lines+markers",
                    name=scen[:30],line=dict(color=_scen_color(scen),width=2)))
            fig_dscr.add_hline(y=1.2,line_dash="dash",line_color=C["coral"],
                annotation_text="1.2× covenant",annotation_font_color=C["coral"])
            fig_dscr.add_hline(y=1.5,line_dash="dot",line_color=C["amber"],
                annotation_text="1.5× threshold",annotation_font_color=C["amber"])
            fig_dscr.update_layout(**_chart_layout("DSCR Stress Trajectory",300))
            _ax_style(fig_dscr)
            st.plotly_chart(fig_dscr,width="stretch")

            st.success("✅ Transition Risk Engine v1.1 executed")
            log_model_run("Transition",{"company":company_name,"sector":sector,
                "pd_max":df_transition["PD_Transition"].max(),"ecl_max":df_transition["ECL_Transition"].max()})
            st.session_state["df_transition_summary"] = df_sum

# ============================================================
# TAB 2 — PHYSICAL RISK  (PHYS-01)
# ============================================================
with physical_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🌍 Physical Risk Engine</h2>",unsafe_allow_html=True)
    st.caption("Asset-level GIS-based physical risk with NGFS-aligned temperature pathways and transparent physical damage assumptions")

    if not st.session_state.get("enable_physical",False):
        st.info("Enable **Physical Risk** in the sidebar.")
    else:
        # Load GIS files only when the Physical Risk module is actually used.
        _gis_missing = _ensure_gis_files()
        if _gis_missing:
            st.warning("⚠️ Some GIS files failed to load. Physical Risk will continue with available layers only.")

        # Public mode: detailed physical methodology notes hidden.
        scope_badge("single", f"Physical KPI cards use the selected reporting year: {REPORTING_YEAR}")

        st.subheader("🏭 Asset Register")
        asset_df = st.data_editor(pd.DataFrame({
            "asset_id":["A1","A2","A3"],
            "asset_type":["Steel Plant","Rolling Mill","Port Logistics Yard"],
            "latitude":[22.80,23.55,20.32],"longitude":[86.20,87.32,86.61],
            "revenue_inr_cr":[3500.0,2000.0,1200.0],
            "base_pd":[0.015,0.015,0.015],"lgd":[0.45,0.45,0.45],
        }),num_rows="fixed",width="stretch",disabled=True)

        fc1,fc2 = st.columns(2)
        EBITDA_MARGIN_PHYS = float(fc1.number_input("EBITDA Margin",value=float(ebitda_margin_0),step=0.01,key="phys_em"))
        INTEREST_PHYS      = float(fc2.number_input("Annual Interest (₹ Cr)",value=float(interest_payment),min_value=1.0,key="phys_int"))
        LGD_PHYS=float(LGD_0); EAD_PHYS=float(exposure_at_default)
        ead_allocation_mode = st.radio(
            "Physical EAD Allocation Mode",
            ["Company-wide proportional exposure", "Full EAD allocated to entered assets"],
            index=0,
            horizontal=True,
            help=("Use proportional mode when the asset register covers only part of company revenue. "
                  "Use full-EAD mode only when the loan is secured against or economically dependent on the listed assets."),
            key="phys_ead_mode",
        )

        # Scenario alignment
        transition_ran_phys = st.session_state.get("transition_ran",False)
        if transition_ran_phys:
            df_t_p = st.session_state.get("df_transition")
            if isinstance(df_t_p,pd.DataFrame) and not df_t_p.empty:
                phys_scenarios = sorted(df_t_p["Scenario"].unique())
                phys_years     = sorted(df_t_p["Year"].unique())
                st.success(f"✅ Aligned with transition scenarios: {phys_scenarios}")
        else:
            phys_scenarios = [BASELINE_SCENARIO]; phys_years = [REPORTING_YEAR]
            st.info("No transition engine run — using baseline scenario only.")

        run_phys = st.button("▶ Run Physical Risk Engine",type="primary",key="run_phys_btn")
        if not run_phys:
            if not render_cached_physical_results():
                st.info("Click **Run Physical Risk Engine** to proceed.")
        else:
            df = asset_df.copy()
            V={"flood":0.6,"heat":0.7,"cyclone":0.6}; KAPPA={"flood":0.4,"heat":0.2,"cyclone":0.5}
            MAX_DOWNTIME={"Steel Plant":90,"Rolling Mill":60,"Port Logistics Yard":45}

            with st.spinner("Extracting hazard data..."):
                # Flood
                # Memory-safe raster access: read only a small window around each asset.
                # This avoids loading the entire GeoTIFF into memory on Streamlit Cloud.
                flood_vals=[]
                try:
                    if rasterio is None or rowcol is None or Window is None:
                        raise RuntimeError(RASTERIO_IMPORT_ERROR or "rasterio unavailable")
                    with rasterio.open(r"Data/floodMapGL_rp100y.tif") as src:
                        pk=max(abs(src.res[0])*111, 1e-6)
                        # Copernicus floodMapGL cell values are water depth in metres.
                        # Use a local p90 depth inside an approx. 1 km window instead of max over 5 km,
                        # because max over large buffers can pick extreme neighbouring river cells.
                        bp=max(1, int(1/pk))
                        for _,a in df.iterrows():
                            try:
                                ri,ci=rowcol(src.transform,a["longitude"],a["latitude"])
                                r0=max(0,ri-bp); r1=min(src.height,ri+bp)
                                c0=max(0,ci-bp); c1=min(src.width,ci+bp)
                                if r1<=r0 or c1<=c0:
                                    flood_vals.append(0.0); continue
                                win=Window(c0,r0,c1-c0,r1-r0)
                                w=src.read(1, window=win, masked=True)
                                v=w.compressed() if hasattr(w, "compressed") else w.ravel()
                                v=v[(v>0)&(v<100)]
                                flood_vals.append(float(np.percentile(v, 90)) if v.size else 0.0)
                            except Exception:
                                flood_vals.append(0.0)
                except Exception as e:
                    st.warning(f"Flood layer skipped: {e}")
                    flood_vals=[0.0]*len(df)
                df["flood_depth_m"]=flood_vals

                # Heat
                try:
                    heat=pd.read_csv(r"Data/era5_test_day_grid.csv")
                    heat["exceed"]=(heat["temp_c"]>=35).astype(int)
                    hg=heat.groupby(["lat","lon"],as_index=False)["exceed"].sum()
                except Exception:
                    hg=pd.DataFrame(columns=["lat","lon","exceed"])
                def nearest_heat(row):
                    if hg.empty: return 0
                    idx=((hg["lat"]-row["latitude"])**2+(hg["lon"]-row["longitude"])**2).idxmin()
                    return int(hg.loc[idx,"exceed"])
                df["heat_days"]=df.apply(nearest_heat,axis=1)

                # Cyclone
                try:
                    cyc=pd.read_csv(
                        r"Data/ibtracs.NI.list.v04r01.csv",
                        usecols=["LAT","LON","USA_WIND"],
                        low_memory=False
                    )
                    cyc=cyc.apply(pd.to_numeric,errors="coerce").dropna()
                    cyc["wind_kmh"]=cyc["USA_WIND"]*1.852
                except Exception as e:
                    st.warning(f"Cyclone layer skipped: {e}")
                    cyc=pd.DataFrame(columns=["LAT","LON","wind_kmh"])
                def max_wind(a):
                    if cyc.empty:
                        return 0.0
                    try:
                        lat1=np.radians(float(a["latitude"]))
                        lon1=np.radians(float(a["longitude"]))
                        lat2=np.radians(cyc["LAT"].astype(float).to_numpy())
                        lon2=np.radians(cyc["LON"].astype(float).to_numpy())
                        dlat=lat2-lat1; dlon=lon2-lon1
                        h=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
                        d=6371*2*np.arcsin(np.sqrt(h))
                        vals=cyc.loc[d<=100,"wind_kmh"]
                        return float(vals.max()) if len(vals)>0 else 0.0
                    except Exception:
                        return 0.0
                df["max_wind_kmh"]=df.apply(max_wind,axis=1)

                df["H_flood"]=df["flood_depth_m"].apply(flood_damage)
                df["H_heat"]=df["heat_days"].apply(heat_damage)
                df["H_cyclone"]=df["max_wind_kmh"].apply(cyclone_damage)
                df["D_total"]=(df["H_flood"]*V["flood"]*KAPPA["flood"]+df["H_heat"]*V["heat"]*KAPPA["heat"]+df["H_cyclone"]*V["cyclone"]*KAPPA["cyclone"]).clip(0,1)
                df["downtime_days"]=df.apply(lambda r: r["D_total"]*MAX_DOWNTIME.get(r["asset_type"],60),axis=1)
                df["revenue_loss"]=df["downtime_days"]/365*df["revenue_inr_cr"]

                TOTAL_REV_LOSS=df["revenue_loss"].sum()
                TOTAL_EBITDA=df["revenue_inr_cr"].sum()*EBITDA_MARGIN_PHYS
                DELTA_EBITDA=TOTAL_REV_LOSS*EBITDA_MARGIN_PHYS
                DSCR_PHYS_BASE=(TOTAL_EBITDA-DELTA_EBITDA)/max(INTEREST_PHYS,1e-6)
                df["PD_Physical"]=(df["base_pd"]*(1+gamma_phys*df["D_total"])).clip(0,1)
                revenue_weight = df["revenue_inr_cr"] / max(df["revenue_inr_cr"].sum(), 1e-6)
                asset_revenue_coverage = float(df["revenue_inr_cr"].sum()/max(revenue_0,1e-6))
                if ead_allocation_mode == "Company-wide proportional exposure":
                    effective_ead_phys = EAD_PHYS * min(max(asset_revenue_coverage, 0.0), 1.0)
                else:
                    effective_ead_phys = EAD_PHYS
                df["EAD_allocated_cr"] = effective_ead_phys * revenue_weight
                df["EAD_allocation_mode"] = ead_allocation_mode
                df["Baseline_ECL_asset"] = df["base_pd"] * df["lgd"] * df["EAD_allocated_cr"]
                df["ECL_asset"] = df["PD_Physical"] * df["lgd"] * df["EAD_allocated_cr"]
                df["Delta_ECL_asset"] = (df["ECL_asset"] - df["Baseline_ECL_asset"]).clip(lower=0)
                PD_PHYS=float((df["PD_Physical"]*revenue_weight).sum())
                PHYSICAL_ECL=df["ECL_asset"].sum(); DELTA_ECL=df["Delta_ECL_asset"].sum()

            # NGFS projection (PHYS-01: transparent illustrative physical damage multiplier)
            df_ngfs_temp = extract_ngfs_temperature_path(df_long,phys_scenarios,REPORTING_YEAR,phys_years)
            df_phys_proj = project_physical_risk_ngfs(
                df_ngfs_temp,df["D_total"].mean(),TOTAL_REV_LOSS,TOTAL_EBITDA,INTEREST_PHYS,
                PD_PHYS,LGD_PHYS,effective_ead_phys,gamma_phys)

            st.session_state["phys_assets"]=df
            st.session_state["phys_summary"]={"Total Revenue Loss (₹ Cr)":TOTAL_REV_LOSS,
                "EBITDA Loss (₹ Cr)":DELTA_EBITDA,"Post-Risk DSCR":DSCR_PHYS_BASE,
                "Physical Risk PD":PD_PHYS,"Physical ECL (₹ Cr)":PHYSICAL_ECL,"ΔECL (₹ Cr)":DELTA_ECL,"Assets Analysed":len(df),"Asset Revenue Coverage %":asset_revenue_coverage*100,"Effective Physical EAD (₹ Cr)":effective_ead_phys,"Physical EAD Allocation Mode":ead_allocation_mode, "Reporting Year": REPORTING_YEAR}
            st.session_state["df_physical_projection"]=df_phys_proj
            st.session_state["physical_ran"]=True
            st.success("✅ Physical Risk Engine v1.2 executed")

            # Summary KPIs — reporting-year snapshot, arranged in 3+2 layout to avoid overlap
            scope_badge("single", f"Asset-level physical risk impact for reporting year {REPORTING_YEAR}")
            render_metric_grid([
                {"title":"Total Revenue Loss", "value":_fmt_money_cr(TOTAL_REV_LOSS), "subtitle":"Estimated annual loss from asset downtime", "accent":C["amber"], "scope":"Reporting year"},
                {"title":"EBITDA Impact", "value":_fmt_money_cr(DELTA_EBITDA), "subtitle":"Revenue loss translated into EBITDA impact", "accent":C["coral"], "scope":"Reporting year"},
                {"title":"Post-Risk DSCR", "value":_fmt_num(DSCR_PHYS_BASE, "×", 2), "subtitle":"Debt service capacity after physical-risk shock", "accent":C["mint"] if DSCR_PHYS_BASE>=1.2 else C["coral"], "scope":"Reporting year"},
                {"title":"Physical Risk PD", "value":_fmt_pct(PD_PHYS), "subtitle":"Portfolio average physical-risk adjusted PD", "accent":C["accent2"], "scope":"Reporting year"},
                {"title":"ΔECL", "value":_fmt_money_cr(DELTA_ECL, 2), "subtitle":"Incremental expected credit loss", "accent":C["purple"], "scope":"Reporting year"},
                {"title":"Physical EAD Used", "value":_fmt_money_cr(effective_ead_phys, 1), "subtitle":ead_allocation_mode, "accent":C["mint"], "scope":"Reporting year"},
            ], columns=3)

            scope_badge("single", f"Asset-level hazard scores for reporting year {REPORTING_YEAR}")
            # Asset vulnerability heatmap
            st.subheader("🗺️ Asset Vulnerability Heatmap")
            score_map={"H_flood":"Flood","H_heat":"Heat","H_cyclone":"Cyclone"}
            hm=df.set_index("asset_id")[[c for c in score_map if c in df.columns]].rename(columns={k:v for k,v in score_map.items() if k in df.columns})*100
            fig_hm=go.Figure(go.Heatmap(z=hm.values.T,x=hm.index.tolist(),y=hm.columns.tolist(),
                colorscale=[[0,C["mint"]],[0.3,C["amber"]],[0.6,"#F97316"],[1.0,C["coral"]]],
                text=[[f"{v:.0f}" for v in row] for row in hm.values.T],texttemplate="%{text}",
                colorbar=dict(title="Risk Score"),zmin=0,zmax=100))
            fig_hm.update_layout(**_chart_layout("Hazard Vulnerability Scores by Asset (0=none, 100=max)",max(220,len(df)*45+80)))
            _ax_style(fig_hm)
            st.plotly_chart(fig_hm,width="stretch")

            # Revenue loss stacked bar
            st.subheader("💸 Revenue Loss by Hazard")
            df["w_flood"] = df["H_flood"] * V["flood"] * KAPPA["flood"]
            df["w_heat"] = df["H_heat"] * V["heat"] * KAPPA["heat"]
            df["w_cyclone"] = df["H_cyclone"] * V["cyclone"] * KAPPA["cyclone"]
            total_w = (df["w_flood"] + df["w_heat"] + df["w_cyclone"]).replace(0, np.nan)
            df["rl_flood"] = (df["revenue_loss"] * df["w_flood"] / total_w).fillna(0)
            df["rl_heat"] = (df["revenue_loss"] * df["w_heat"] / total_w).fillna(0)
            df["rl_cyclone"] = (df["revenue_loss"] * df["w_cyclone"] / total_w).fillna(0)
            fig_rev=go.Figure()
            for haz,col,clr in [("rl_flood","Flood",C["accent3"]),("rl_heat","Heat",C["amber"]),("rl_cyclone","Cyclone",C["coral"])]:
                if haz in df.columns:
                    fig_rev.add_trace(go.Bar(x=df["asset_id"],y=df[haz].fillna(0),name=col,marker_color=clr))
            fig_rev.update_layout(**_chart_layout("Revenue Loss Decomposition by Hazard (₹ Cr)",300),barmode="stack")
            _ax_style(fig_rev); st.plotly_chart(fig_rev,width="stretch")

            # NGFS scenario fan charts (PHYS-01)
            if not df_phys_proj.empty:
                scope_badge("multi", "Forward-looking physical-risk projection across the same scenario years as transition risk")
                st.subheader("🌡️ NGFS Physical Risk Projections")
                st.caption("Revenue loss under each NGFS scenario · illustrative damage multiplier · P10/P50/P90 uncertainty")
                tabs_scen = st.tabs(sorted(df_phys_proj["Scenario"].unique()))
                for tab_s,scen in zip(tabs_scen,sorted(df_phys_proj["Scenario"].unique())):
                    with tab_s:
                        ds=df_phys_proj[df_phys_proj["Scenario"]==scen].sort_values("Year")
                        fig_f=make_subplots(rows=2,cols=2,
                            subplot_titles=("Revenue Loss — P10/P50/P90","Physical PD","DSCR Under Physical Stress","Chronic vs Acute Split"),
                            vertical_spacing=0.18,horizontal_spacing=0.12)
                        # Fan
                        fig_f.add_trace(go.Scatter(
                            x=ds["Year"].tolist()+ds["Year"].tolist()[::-1],
                            y=ds["Revenue_Loss_P90_Cr"].tolist()+ds["Revenue_Loss_P10_Cr"].tolist()[::-1],
                            fill="toself",fillcolor="rgba(6,182,212,0.15)",line=dict(color="rgba(0,0,0,0)"),name="P10–P90"),row=1,col=1)
                        fig_f.add_trace(go.Scatter(x=ds["Year"],y=ds["Revenue_Loss_P50_Cr"],mode="lines+markers",
                            line=dict(color=C["accent2"],width=2.5),name="P50"),row=1,col=1)
                        fig_f.add_trace(go.Scatter(x=ds["Year"],y=ds["PD_Physical"],mode="lines+markers",
                            line=dict(color=C["coral"],width=2.5),name="Physical PD"),row=1,col=2)
                        fig_f.add_trace(go.Scatter(x=ds["Year"],y=ds["DSCR_Physical"],mode="lines+markers",
                            line=dict(color=C["amber"],width=2.5),name="DSCR"),row=2,col=1)
                        fig_f.add_hline(y=1.2,line_dash="dash",line_color=C["coral"],annotation_text="1.2× covenant",row=2,col=1)
                        fig_f.add_trace(go.Bar(x=ds["Year"],y=ds["Chronic_Loss_Cr"],name="Chronic",marker_color=C["accent3"]),row=2,col=2)
                        fig_f.add_trace(go.Bar(x=ds["Year"],y=ds["Acute_Loss_Cr"],name="Acute",marker_color=C["coral"]),row=2,col=2)
                        fig_f.update_layout(**_chart_layout(f"{scen} — Physical Risk Projections",480),barmode="stack")
                        _ax_style(fig_f,rows=2,cols=2); st.plotly_chart(fig_f,width="stretch")
                        wr=ds.loc[ds["PD_Physical"].idxmax()]
                        c1p,c2p,c3p,c4p=st.columns(4)
                        c1p.metric("Worst Year",f"{int(wr['Year'])}"); c2p.metric("Peak Rev Loss",f"₹{wr['Revenue_Loss_P50_Cr']:.1f} Cr")
                        c3p.metric("Peak Physical PD",f"{wr['PD_Physical']:.2%}"); c4p.metric("Temp Anomaly",f"+{wr['Delta_T_C']:.2f}°C")

            log_model_run("Physical",{"company":company_name,"pd_physical":PD_PHYS,"delta_ecl":DELTA_ECL})

# ============================================================
# TAB 3 — TARGETS
# ============================================================
with targets_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🎯 Transition & BRSR Targets</h2>",unsafe_allow_html=True)
    st.caption("Management-defined decarbonisation targets — financial impact across transition risk and BRSR compliance trajectory")
    if not st.session_state.get("enable_targets",False):
        st.info("Enable **Transition Targets** in the sidebar.")
    elif not st.session_state.get("transition_ran",False):
        st.warning("Run Transition Risk Engine first.")
    else:
        df_base=st.session_state.get("df_transition")
        if not isinstance(df_base,pd.DataFrame) or df_base.empty:
            st.warning("Transition results unavailable.")
        else:
            # ── SECTION A: FINANCIAL TARGETS ──
            st.markdown(f"<h3 style='color:{C['accent2']};'>A · Financial Transition Targets</h3>",unsafe_allow_html=True)
            st.caption("Apply emissions reduction, revenue, and margin adjustments by time horizon.")
            tc1,tc2,tc3=st.columns(3)
            with tc1:
                st.markdown(f"<div style='color:{C['accent2']};font-weight:600;margin-bottom:8px;'>Short-Term (≤2030)</div>",unsafe_allow_html=True)
                s_em=st.slider("Emissions Reduction (%)",0,100,0,key="s_em")
                s_rev=st.slider("Revenue Adj (%)",-30,30,0,key="s_rev")
                s_mar=st.slider("Margin Adj (%)",-30,30,0,key="s_mar")
            with tc2:
                st.markdown(f"<div style='color:{C['amber']};font-weight:600;margin-bottom:8px;'>Medium-Term (2031–2040)</div>",unsafe_allow_html=True)
                m_em=st.slider("Emissions Reduction (%)",0,100,0,key="m_em")
                m_rev=st.slider("Revenue Adj (%)",-30,30,0,key="m_rev")
                m_mar=st.slider("Margin Adj (%)",-30,30,0,key="m_mar")
            with tc3:
                st.markdown(f"<div style='color:{C['coral']};font-weight:600;margin-bottom:8px;'>Long-Term (>2040)</div>",unsafe_allow_html=True)
                l_em=st.slider("Emissions Reduction (%)",0,100,0,key="l_em")
                l_rev=st.slider("Revenue Adj (%)",-30,30,0,key="l_rev")
                l_mar=st.slider("Margin Adj (%)",-30,30,0,key="l_mar")

            st.divider()

            # ── SECTION B: BRSR OPERATIONAL TARGETS ──
            st.markdown(f"<h3 style='color:{C['mint']};'>B · BRSR Operational Targets</h3>",unsafe_allow_html=True)
            st.caption("Set SEBI BRSR Core improvement commitments. These reduce BRSR risk overlay as targets are met, feeding into Integrated Risk.")

            brsr_ran_t = st.session_state.get("brsr_ran", False)
            current_brsr_pd = st.session_state.get("brsr_pd_adj", 0.0)
            current_readiness = 0.0
            current_flags = []
            if brsr_ran_t:
                bs_t = st.session_state.get("brsr_summary")
                if isinstance(bs_t, pd.DataFrame) and not bs_t.empty:
                    current_readiness = float(bs_t.iloc[0].get("Readiness_Score", 0))
                bf_t = st.session_state.get("brsr_flags")
                if isinstance(bf_t, pd.DataFrame) and "Flag" in bf_t.columns:
                    current_flags = bf_t["Flag"].tolist()

            if brsr_ran_t:
                st.markdown(f"**Current state:** BRSR Governance Signal = **+{current_brsr_pd*10000:.0f} bps** · Readiness = **{current_readiness:.0f}%** · Active flags: **{len(current_flags)}**")
            else:
                st.info("Run **BRSR Diagnostics** first to enable BRSR target planning.")

            bc1, bc2 = st.columns(2)
            with bc1:
                st.markdown(f"<div style='color:{C['mint']};font-weight:600;margin-bottom:8px;'>Near-Term BRSR Actions (by 2027)</div>",unsafe_allow_html=True)
                tgt_renewable   = st.slider("Target Renewable Energy Share (%)", 0, 100, min(100, int(st.session_state.get("brsr_summary",pd.DataFrame()).iloc[0].get("Renewable_%",15) if brsr_ran_t and not st.session_state.get("brsr_summary",pd.DataFrame()).empty else 15) + 10), key="tgt_ren")
                tgt_coverage    = st.slider("Target Emissions Coverage (%)", 0, 100, min(100, int(st.session_state.get("brsr_summary",pd.DataFrame()).iloc[0].get("Target_Coverage_%",50) if brsr_ran_t and not st.session_state.get("brsr_summary",pd.DataFrame()).empty else 50) + 15), key="tgt_cov")
                tgt_hazwaste    = st.slider("Target Hazardous Waste (%)", 0, 100, max(0, int(st.session_state.get("brsr_summary",pd.DataFrame()).iloc[0].get("Hazardous_Waste_%",20) if brsr_ran_t and not st.session_state.get("brsr_summary",pd.DataFrame()).empty else 20) - 5), key="tgt_hw")
            with bc2:
                st.markdown(f"<div style='color:{C['amber']};font-weight:600;margin-bottom:8px;'>Governance Commitments</div>",unsafe_allow_html=True)
                tgt_board       = st.checkbox("Commit Board Climate Oversight", value=False, key="tgt_board")
                tgt_netzero     = st.checkbox("Commit Long-term Climate Target", value=False, key="tgt_nz")
                tgt_verified    = st.checkbox("Commit to Third-Party Verification", value=False, key="tgt_vd")
                tgt_scope3      = st.checkbox("Commit Scope 3 Disclosure", value=True, key="tgt_s3")

            # Compute BRSR target risk overlay reduction
            def _compute_brsr_target_pd(flags_in, tgt_ren, tgt_cov, tgt_hw, tgt_board, tgt_nz, tgt_vd, tgt_s3):
                remaining_flags = []
                for f in flags_in:
                    if f == "Low renewable energy (<20%)" and tgt_ren >= 20: continue
                    if f == "Low target coverage (<50%)"  and tgt_cov >= 50: continue
                    if f == "High hazardous waste (>30%)" and tgt_hw <= 30:  continue
                    if f == "No verified emissions data"  and tgt_vd:         continue
                    if f == "Missing Scope 3 disclosure"  and tgt_s3:         continue
                    remaining_flags.append(f)
                new_overlay = float(np.clip(sum(BRSR_PD_SPREAD.get(f,0) for f in remaining_flags), 0, 0.015))
                return new_overlay, remaining_flags

            brsr_target_overlay, brsr_remaining_flags = _compute_brsr_target_pd(
                current_flags, tgt_renewable, tgt_coverage, tgt_hazwaste,
                tgt_board, tgt_netzero, tgt_verified, tgt_scope3
            )
            brsr_pd_reduction = current_brsr_pd - brsr_target_overlay

            # Preview BRSR target impact
            if brsr_ran_t:
                tp1, tp2, tp3 = st.columns(3)
                tp1.metric("Current BRSR Uplift", f"+{current_brsr_pd*10000:.0f} bps")
                tp2.metric("Target BRSR Uplift",  f"+{brsr_target_overlay*10000:.0f} bps",
                           delta=f"-{brsr_pd_reduction*10000:.0f} bps", delta_color="normal")
                tp3.metric("Flags Resolved",
                           f"{len(current_flags) - len(brsr_remaining_flags)} / {len(current_flags)}")

            if st.button("▶ Run Target Scenario", type="primary"):
                def apply_t(row):
                    if row["Year"]<=2030: em,rv,mr=s_em,s_rev,s_mar
                    elif row["Year"]<=2040: em,rv,mr=m_em,m_rev,m_mar
                    else: em,rv,mr=l_em,l_rev,l_mar
                    rt=row["Revenue"]*(1+rv/100); mt=row["EBITDA_Margin"]*(1+mr/100)
                    pt=np.clip(max(base_pd, row["PD_Transition"]*(1-em/100)),0,1)
                    et=ecl_cr(pt, row["LGD"], exposure_at_default)
                    return pd.Series({"Revenue_Target":rt,"EBITDA_Target":rt*mt,"PD_Target":pt,"ECL_Target":et})
                df_tgt=pd.concat([df_base[["Scenario","Year"]],df_base.apply(apply_t,axis=1)],axis=1).reset_index(drop=True)
                st.session_state["df_target"]=df_tgt; st.session_state["targets_ran"]=True

                # Store BRSR target results
                st.session_state["brsr_target_overlay"]   = brsr_target_overlay
                st.session_state["brsr_pd_reduction"]    = brsr_pd_reduction
                st.session_state["brsr_remaining_flags"] = brsr_remaining_flags

                eff=(df_tgt.groupby("Scenario").agg({"PD_Target":"max"}).rename(columns={"PD_Target":"PD_Target_Max"}).reset_index()
                     .merge(df_base.groupby("Scenario").agg({"PD_Transition":"max"}).rename(columns={"PD_Transition":"PD_Base_Max"}).reset_index(),on="Scenario"))
                eff["PD_Reduction_%"]=((eff["PD_Base_Max"]-eff["PD_Target_Max"])/eff["PD_Base_Max"]*100).round(2)
                st.session_state["df_target_effect"]=eff

                st.success("✅ Targets applied — BRSR target overlay saved to Integrated Risk tab")

                # ── Results display ──
                col_r1, col_r2 = st.columns([1, 1])
                with col_r1:
                    st.markdown(f"<h4 style='color:{C['accent2']};'>Financial Target Effectiveness</h4>",unsafe_allow_html=True)
                    st.dataframe(eff.style.format({"PD_Target_Max":"{:.3%}","PD_Base_Max":"{:.3%}","PD_Reduction_%":"{:.1f}%"})
                        .background_gradient(subset=["PD_Reduction_%"],cmap="Greens"),width="stretch",hide_index=True)

                with col_r2:
                    if brsr_ran_t and len(current_flags) > 0:
                        st.markdown(f"<h4 style='color:{C['mint']};'>BRSR Governance Signal — Before vs After Targets</h4>",unsafe_allow_html=True)
                        brsr_bar_labels = ["Current Uplift", "Target Uplift"]
                        brsr_bar_vals   = [current_brsr_pd*10000, brsr_target_overlay*10000]
                        fig_brsr_tgt = go.Figure(go.Bar(
                            x=brsr_bar_labels, y=brsr_bar_vals,
                            marker_color=[C["coral"], C["mint"]],
                            text=[f"{v:.0f} bps" for v in brsr_bar_vals], textposition="outside",
                        ))
                        fig_brsr_tgt.update_layout(**_chart_layout("BRSR Governance Signal Reduction (bps)", 260),
                            yaxis_title="Basis Points")
                        _ax_style(fig_brsr_tgt)
                        st.plotly_chart(fig_brsr_tgt, width="stretch")

                # Baseline vs target PD chart
                df_plot=df_base.merge(df_tgt,on=["Scenario","Year"],how="inner")
                fig_tgt=go.Figure()
                for scen in df_plot["Scenario"].unique():
                    ds=df_plot[df_plot["Scenario"]==scen]; clr=_scen_color(scen)
                    fig_tgt.add_trace(go.Scatter(x=ds["Year"],y=ds["PD_Transition"],mode="lines",
                        name=f"{scen[:20]} Baseline",line=dict(dash="dash",color=clr,width=1.5)))
                    fig_tgt.add_trace(go.Scatter(x=ds["Year"],y=ds["PD_Target"],mode="lines+markers",
                        name=f"{scen[:20]} + Financial Targets",line=dict(color=clr,width=2.5),marker=dict(size=6)))
                # Show integrated target line (transition PD target + BRSR target overlay)
                if brsr_ran_t:
                    for scen in df_tgt["Scenario"].unique():
                        ds = df_tgt[df_tgt["Scenario"]==scen]
                        pd_integ_target = np.clip(ds["PD_Target"] + brsr_target_overlay, PD_FLOOR, PD_CAP)
                        fig_tgt.add_trace(go.Scatter(x=ds["Year"], y=pd_integ_target, mode="lines",
                            name=f"{scen[:20]} + BRSR Target",
                            line=dict(color=C["mint"], width=1.5, dash="dot")))
                fig_tgt.update_layout(**_chart_layout("PD — Baseline vs Financial Target vs Integrated (+ BRSR) Target", 340))
                fig_tgt.update_yaxes(tickformat=".1%")
                _ax_style(fig_tgt)
                st.plotly_chart(fig_tgt, width="stretch")

                if brsr_ran_t:
                    st.caption(f"**BRSR note:** Target renewable share {tgt_renewable}%, target coverage {tgt_coverage}%, target hazardous waste {tgt_hazwaste}%. These resolve {len(current_flags)-len(brsr_remaining_flags)} of {len(current_flags)} current BRSR flags, reducing risk overlay from +{current_brsr_pd*10000:.0f} bps to +{brsr_target_overlay*10000:.0f} bps. The dotted 'BRSR Target' line includes this reduction.")
            else:
                if not render_cached_targets_results():
                    st.info("Set targets above and click **Run Target Scenario**.")

# ============================================================
# TAB 4 — INTEGRATED RISK  (INT-01)
# ============================================================
with integrated_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🧩 Integrated Climate Risk</h2>",unsafe_allow_html=True)
    st.caption("Gaussian copula joint PD · BRSR overlay layer · Capital assessment · ICAAP metrics")

    transition_ran_i = st.session_state.get("transition_ran",False)
    physical_ran_i   = st.session_state.get("physical_ran",False)
    physical_enabled = st.session_state.get("enable_physical",False)
    brsr_ran_i       = st.session_state.get("brsr_ran",False)

    if physical_enabled and physical_ran_i and transition_ran_i: exec_mode="combined"
    elif transition_ran_i: exec_mode="transition_only"
    elif physical_enabled and physical_ran_i: exec_mode="physical_only"
    else: exec_mode="none"

    if exec_mode == "none":
        st.warning("Run Transition Risk or Physical Risk first.")
    else:

        correlation = st.slider("Transition–Physical Correlation (ρ)",0.0,0.6,default_correlation,0.05,key="corr_sl") if exec_mode=="combined" else default_correlation

        # ── BRSR overlay — must be computed BEFORE the mode banner that references _brsr_label ──
        # Use target overlay if targets have been run, else use current overlay
        brsr_pd_adj  = 0.0
        _brsr_label  = "not run"        # safe default — overwritten below if BRSR ran
        if brsr_ran_i:
            targets_ran_i = st.session_state.get("targets_ran", False)
            if targets_ran_i and "brsr_target_overlay" in st.session_state:
                brsr_pd_adj = float(st.session_state["brsr_target_overlay"])
                _brsr_label = "after targets"
            else:
                brsr_pd_adj = float(st.session_state.get("brsr_pd_adj", 0.0))
                _brsr_label = "current"

        scope_badge("multi", "Integrated risk results use multi-year scenario peak/worst transition stress, plus reporting-year physical/BRSR signals where available.")

        # Mode banner — all active layers now that _brsr_label is defined
        _layers = []
        if exec_mode == "combined":       _layers.append("🔗 Transition + Physical (Copula)")
        elif exec_mode == "transition_only": _layers.append("⚡ Transition Only")
        elif exec_mode == "physical_only":   _layers.append("🌍 Physical Only")
        if brsr_ran_i:
            _brsr_tag = f"📘 BRSR (+{brsr_pd_adj*10000:.0f}bps, {_brsr_label})"
            _layers.append(_brsr_tag)
        if st.session_state.get("targets_ran", False):
            _layers.append("🎯 Targets Applied")
        _mode_str = " &nbsp;+&nbsp; ".join(_layers)
        st.markdown(f"<div style='background:{C['card']};border-left:4px solid {C['accent2']};border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:13px;color:{C['accent2']};font-weight:600;'>{_mode_str}</div>",unsafe_allow_html=True)

        # Collect results
        pd_trans=ecl_trans=dscr_trans=stranded_t=capex_t=None
        if transition_ran_i:
            df_ti=st.session_state.get("df_transition")
            if isinstance(df_ti,pd.DataFrame) and not df_ti.empty:
                pd_trans=float(df_ti["PD_Transition"].max()); ecl_trans=float(df_ti["ECL_Transition"].max())
                dscr_trans=float(df_ti["DSCR"].min()); stranded_t=float(df_ti["Stranded_Assets"].max()); capex_t=float(df_ti["CAPEX_Gap"].max())

        pd_phys=ecl_phys=dscr_phys=rev_loss_phys=None
        if exec_mode in ["physical_only","combined"]:
            ps=st.session_state.get("phys_summary")
            if isinstance(ps,dict):
                pd_phys=float(ps.get("Physical Risk PD",0)); ecl_phys=float(ps.get("ΔECL (₹ Cr)",0))
                dscr_phys=float(ps.get("Post-Risk DSCR",1.5)); rev_loss_phys=float(ps.get("Total Revenue Loss (₹ Cr)",0))

        # Physical PD aligned with the integrated stress horizon.
        # Integrated transition metrics use peak/worst path values, so physical risk should not be
        # pulled from an arbitrary first scenario at the reporting year. Use the larger of the
        # reporting-year asset snapshot and the physical projection peak.
        pd_phys_scen=pd_phys
        df_pp=st.session_state.get("df_physical_projection")
        if isinstance(df_pp,pd.DataFrame) and not df_pp.empty and "PD_Physical" in df_pp.columns:
            pd_phys_scen=max(float(pd_phys or 0.0), float(df_pp["PD_Physical"].max()))
        if pd_phys_scen is not None: pd_phys_scen=float(np.clip(pd_phys_scen,0,1))

        # Integrated borrower PD: use union probability, not only joint/intersection probability.
        pd_copula=None
        joint_intersection_pd=0.0
        if exec_mode=="combined" and pd_trans is not None and pd_phys_scen is not None:
            pd_copula, joint_intersection_pd = combined_pd_union(pd_trans, pd_phys_scen, correlation)
        elif transition_ran_i and pd_trans is not None:
            pd_copula=pd_trans
        elif pd_phys_scen is not None:
            pd_copula=pd_phys_scen

        EAD=float(exposure_at_default)
        # Apply BRSR as bounded governance multiplier to reduce double-counting.
        brsr_mult = brsr_governance_multiplier(brsr_pd_adj)
        pd_integrated=float(np.clip(pd_copula*brsr_mult,PD_FLOOR,PD_CAP)) if pd_copula is not None else None
        lgd_integrated = float(np.clip(max(LGD_0, float(df_ti["LGD"].max()) if transition_ran_i and isinstance(df_ti, pd.DataFrame) and not df_ti.empty else LGD_0), 0, 1))
        ecl_integrated=ecl_cr(pd_integrated, lgd_integrated, EAD) if pd_integrated is not None else 0
        dscr_integrated=min(d for d in [dscr_trans,dscr_phys] if d is not None) if any(d is not None for d in [dscr_trans,dscr_phys]) else None

        if pd_integrated is None:
            st.warning("Integrated PD could not be computed. Check module results.")

        ecl_ead_ratio=ecl_integrated/EAD if EAD>0 else 0

        # Risk score (0-100)
        pd_score  =min(100,(pd_integrated/0.20)*100)
        ecl_score =min(100,(ecl_ead_ratio/0.05)*100)
        dscr_score=100 if (dscr_integrated and dscr_integrated<1.0) else 60 if (dscr_integrated and dscr_integrated<1.2) else 30 if (dscr_integrated and dscr_integrated<1.5) else 10
        risk_score_int=round(0.4*pd_score+0.4*ecl_score+0.2*dscr_score,1)

        if risk_score_int<30: rag="🟢"; rag_color=C["mint"]; rag_text="Low Climate Risk"
        elif risk_score_int<60: rag="🟡"; rag_color=C["amber"]; rag_text="Moderate Climate Risk"
        elif risk_score_int<80: rag="🟠"; rag_color="#F97316"; rag_text="Severe Climate Risk"
        else: rag="🔴"; rag_color=C["coral"]; rag_text="Extreme Climate Risk"

        # RAG banner
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,{C['card']},{C['bg_mid']});border:1px solid {rag_color};border-radius:10px;
                    padding:16px 20px;margin-bottom:16px;display:flex;align-items:center;gap:16px;">
          <div style="font-size:36px;">{rag}</div>
          <div>
            <div style="font-size:18px;font-weight:700;color:{rag_color};">{rag_text}</div>
            <div style="font-size:13px;color:{C['slate']};margin-top:3px;">
              Integrated Risk Score: {risk_score_int}/100 · PD: {pd_integrated:.2%} · ECL/EAD: {ecl_ead_ratio:.2%}
              {f' · BRSR factor: ×{brsr_mult:.2f}' if brsr_ran_i and brsr_pd_adj>0 else ''}
            </div>
          </div>
        </div>""",unsafe_allow_html=True)

        # KPI grid
        ik1,ik2,ik3,ik4,ik5=st.columns(5)
        ik1.metric("Integrated PD",f"{pd_integrated:.2%}",help="Union-style combined PD × bounded BRSR governance factor")
        ik2.metric("Integrated ECL",f"₹{ecl_integrated:,.1f} Cr")
        ik3.metric("ECL / EAD",f"{ecl_ead_ratio:.2%}")
        ik4.metric("Min DSCR",f"{dscr_integrated:.2f}×" if dscr_integrated else "—")
        ik5.metric("Risk Score",f"{risk_score_int}/100")

        if brsr_ran_i and brsr_pd_adj>0:
            st.markdown(f"<div style='color:{C['slate']};font-size:12px;margin-top:4px;'>BRSR governance quality has been incorporated as a bounded multiplier in the integrated risk score.</div>", unsafe_allow_html=True)

        st.divider()

        # ── RISK DECOMPOSITION ──
        st.subheader("🧩 Risk Decomposition")
        dec_col1, dec_col2 = st.columns([1,2])
        with dec_col1:
            sources,values=[],[]
            if transition_ran_i and ecl_trans: sources.append("Transition"); values.append(ecl_trans)
            if exec_mode in ["physical_only","combined"] and ecl_phys: sources.append("Physical"); values.append(ecl_phys)
            if brsr_ran_i and brsr_pd_adj>0:
                brsr_ecl_equiv=max(pd_integrated-(pd_copula or 0),0)*lgd_integrated*EAD; sources.append("BRSR Operational"); values.append(brsr_ecl_equiv)
            if sources:
                fig_dec=go.Figure(go.Pie(labels=sources,values=values,
                    hole=0.45,marker=dict(colors=[C["accent2"],C["coral"],C["purple"]][:len(sources)]),
                    textinfo="label+percent",hovertemplate="%{label}: ₹%{value:.1f} Cr<extra></extra>"))
                fig_dec.update_layout(**_chart_layout("ECL Risk Decomposition",280),showlegend=False,
                    annotations=[dict(text=f"₹{ecl_integrated:.1f}Cr",x=0.5,y=0.5,font_size=14,font_color=C["white"],showarrow=False)])
                st.plotly_chart(fig_dec,width="stretch")
        with dec_col2:
            if transition_ran_i:
                df_ti3=st.session_state.get("df_transition")
                if isinstance(df_ti3,pd.DataFrame) and not df_ti3.empty:
                    fig_sc=go.Figure()
                    for scen in df_ti3["Scenario"].unique():
                        ds=df_ti3[df_ti3["Scenario"]==scen]
                        fig_sc.add_trace(go.Bar(x=["PD","ECL/EAD","DSCR Stress"],
                            y=[ds["PD_Transition"].max()*100,ds["ECL_Transition"].max()/EAD*100,max(0,(1.5-ds["DSCR"].min())/1.5*100)],
                            name=scen[:25],marker_color=_scen_color(scen)))
                    fig_sc.update_layout(**_chart_layout("Risk Metrics by Scenario (normalised %)",280),barmode="group",yaxis_title="%")
                    _ax_style(fig_sc); st.plotly_chart(fig_sc,width="stretch")
        # Public mode: proprietary integration working is hidden.

        # ── SCENARIO STRESS TABLE ──
        st.subheader("📊 Full Scenario Stress Matrix")
        if transition_ran_i:
            df_ti4=st.session_state.get("df_transition")
            if isinstance(df_ti4,pd.DataFrame) and not df_ti4.empty:
                df_stress=(df_ti4.groupby("Scenario").agg({
                    "PD_Transition":"max","ECL_Transition":"max","DSCR":"min",
                    "Carbon_Burden":"max","EBITDA_Margin":"min",
                    "Stranded_Assets":"max","CAPEX_Gap":"max",
                }).round(4).reset_index().rename(columns={
                    "PD_Transition":"Peak PD","ECL_Transition":"Peak ECL (₹Cr)",
                    "DSCR":"Min DSCR","Carbon_Burden":"Max Carbon Burden",
                    "EBITDA_Margin":"Min EBITDA Margin","Stranded_Assets":"Stranded (₹Cr)","CAPEX_Gap":"CAPEX Gap / Surplus (₹Cr)"
                }))
                st.dataframe(df_stress.style
                    .format({"Peak PD":"{:.3%}","Peak ECL (₹Cr)":"{:.1f}","Min DSCR":"{:.2f}",
                             "Max Carbon Burden":"{:.2%}","Min EBITDA Margin":"{:.2%}",
                             "Stranded (₹Cr)":"{:.0f}","CAPEX Gap / Surplus (₹Cr)":"{:.0f}"})
                    .background_gradient(subset=["Peak PD"],cmap="Reds")
                    .background_gradient(subset=["Min DSCR"],cmap="RdYlGn"),
                    width="stretch",hide_index=True)

        # ── ICAAP SUMMARY ──
        st.subheader("🏦 ICAAP Capital Assessment")
        capital_signal="Severe" if ecl_ead_ratio>=0.05 else "Elevated" if ecl_ead_ratio>=0.03 else "Moderate" if ecl_ead_ratio>=0.015 else "Limited"
        cap_color=C["coral"] if capital_signal in ["Severe","Elevated"] else C["amber"] if capital_signal=="Moderate" else C["mint"]
        cap1,cap2,cap3,cap4=st.columns(4)
        cap1.metric("Capital Signal",capital_signal)
        cap2.metric("Unexpected Climate Loss",f"₹{max(0,ecl_integrated-ecl_integrated*0.7):.1f} Cr",help="95th pct - mean approximation")
        cap3.metric("Climate Capital (% EAD)",f"{ecl_ead_ratio:.2%}")
        cap4.metric("ISSB S2 Reference","§15(b), §16")
        st.markdown(f"<div style='background:{C['card']};border-left:3px solid {cap_color};padding:10px 14px;border-radius:6px;font-size:13px;color:{cap_color};font-weight:600;margin-top:8px;'>Capital Stress: {capital_signal} Climate Capital Impact — {'Strategic intervention required.' if capital_signal in ['Severe','Elevated'] else 'Monitoring and capital buffer planning recommended.' if capital_signal=='Moderate' else 'Within normal tolerance.'}</div>",unsafe_allow_html=True)

        # ── ISSB S2 SUMMARY TABLE ──
        st.subheader("📋 ISSB S2–Aligned Integrated Summary")
        df_isummary=pd.DataFrame([
            {"Metric":"Integrated PD (Union PD × BRSR factor)","Value":f"{pd_integrated:.4f}","ISSB S2":"§15(a)"},
            {"Metric":"Integrated ECL (₹ Cr)","Value":f"{ecl_integrated:.2f}","ISSB S2":"§15(b)"},
            {"Metric":"Post-Stress DSCR","Value":f"{dscr_integrated:.2f}" if dscr_integrated else "—","ISSB S2":"§15(c)"},
            {"Metric":"ECL / EAD","Value":f"{ecl_ead_ratio:.4f}","ISSB S2":"§16"},
            {"Metric":"BRSR Governance Factor","Value":f"x{brsr_mult:.3f}","ISSB S2":"§15(a) operational"},
            {"Metric":"Capital Stress Signal","Value":capital_signal,"ISSB S2":"§16"},
            {"Metric":"Integrated Risk Score (0–100)","Value":f"{risk_score_int}","ISSB S2":"§14–16"},
            {"Metric":"Stranded Assets (₹ Cr)","Value":f"{stranded_t:.0f}" if stranded_t else "—","ISSB S2":"§22"},
            {"Metric":"CAPEX Gap / Surplus (₹ Cr)","Value":_fmt_capex_position(capex_t) if capex_t is not None else "—","ISSB S2":"§14"},
        ])
        st.dataframe(df_isummary.style.set_properties(**{"background-color":C["bg_dark"],"color":C["text"]}),
            width="stretch",hide_index=True)

        st.session_state["df_integrated_summary"]=df_isummary
        st.session_state["integrated_ran"]=True
        st.session_state["model_inputs"]={"company":company_name,"sector":sector,"reporting_year":REPORTING_YEAR,
            "revenue":revenue_0,"ebitda_margin":ebitda_margin_0,"interest_expense":interest_payment,
            "total_emissions":TOTAL_EMISSIONS,"exposure_at_default":EAD,"base_pd":base_pd,"lgd":LGD_0}

        # ── MONTE CARLO ──
        st.divider()
        st.subheader("🎲 Monte Carlo Climate Stress")
        run_mc=st.checkbox("Run Monte Carlo Simulation")
        if run_mc:
            n_sims=st.slider("Simulations",100,5000,1000,100)
            df_tmc=st.session_state.get("df_transition")
            if isinstance(df_tmc,pd.DataFrame) and not df_tmc.empty:
                wr=df_tmc.loc[df_tmc["PD_Transition"].idxmax()]
                cb_mc=wr["Carbon_Burden"]; pl_mc=max(0,1-wr["EBITDA_Margin"]/max(ebitda_margin_0,1e-6)); cp_mc=cb_mc*revenue_0/max(TOTAL_EMISSIONS,1)
            else: cb_mc=0.05; pl_mc=0.02; cp_mc=50.0

            pd_sim=[]; ecl_sim=[]
            for _ in range(n_sims):
                sh_c,sh_p,sh_g=np.random.multivariate_normal([0,0,0],CLIMATE_DRIVER_COV)
                cps=cp_mc*np.exp(MC_CARBON_VOL*sh_c); pds=pl_mc*np.exp(sh_p)
                cc=(TOTAL_EMISSIONS*cps*USD_INR/1e7)*(1-carbon_pass_through)
                rv=revenue_0*(1+gdp_sensitivity*sh_g)*(1-pds)
                rv=max(rv, revenue_0*0.05)
                eb=rv*ebitda_margin_0-cc; cbu=cc/max(rv,1)
                cs=BASE_CREDIT_SPREAD*(1+spread_beta*cbu)
                ds=eb/max(interest_payment*(1+cs),1e-6)
                dg=np.clip(1.5-ds,-4.0,6.0)
                lp=logit(base_pd)+alpha_dscr*dg+beta_carbon_credit*cbu
                pt=np.clip(max(base_pd, sigmoid(lp)),PD_FLOOR,PD_CAP)
                pp=np.clip(pd_phys_scen*(1+pds),0,1) if physical_ran_i and pd_phys_scen else 0
                pj=combined_pd_union(pt,pp,correlation)[0] if physical_ran_i else pt
                pj=float(np.clip(pj*brsr_governance_multiplier(brsr_pd_adj),PD_FLOOR,PD_CAP))
                lj=np.clip(LGD_0*(1+0.2*cbu+LGD_PHYSICAL_MULTIPLIER*pds),0,1)
                pd_sim.append(pj); ecl_sim.append(ecl_cr(pj, lj, EAD))

            pd_sim=np.array(pd_sim); ecl_sim=np.array(ecl_sim)
            pd_mean=pd_sim.mean(); pd_95=np.percentile(pd_sim,95)
            ecl_mean=ecl_sim.mean(); ecl_95=np.percentile(ecl_sim,95)
            ecl_es=ecl_sim[ecl_sim>=ecl_95].mean(); ul=ecl_95-ecl_mean

            mc1,mc2,mc3,mc4=st.columns(4)
            mc1.metric("Mean PD",f"{pd_mean:.2%}"); mc2.metric("PD 95th",f"{pd_95:.2%}")
            mc3.metric("Climate VaR (₹Cr)",f"{ecl_95:,.2f}"); mc4.metric("Unexpected Loss (₹Cr)",f"{ul:,.2f}")

            fig_mc=make_subplots(rows=1,cols=2,subplot_titles=("ECL Distribution","PD vs ECL Scatter"))
            fig_mc.add_trace(go.Histogram(x=ecl_sim,nbinsx=40,marker_color=C["accent2"],opacity=0.75,name="ECL"),row=1,col=1)
            fig_mc.add_vline(x=ecl_95,line_dash="dash",line_color=C["coral"],annotation_text=f"VaR 95%: {ecl_95:.1f}",row=1,col=1)
            fig_mc.add_trace(go.Scatter(x=pd_sim,y=ecl_sim,mode="markers",marker=dict(color=C["accent3"],size=3,opacity=0.3),name="Simulation"),row=1,col=2)
            fig_mc.update_layout(**_chart_layout("Monte Carlo Results",320)); _ax_style(fig_mc,rows=1,cols=2)
            st.plotly_chart(fig_mc,width="stretch")
            st.session_state["mc_results"]={"Mean_PD":pd_mean,"PD_95":pd_95,"Mean_ECL":ecl_mean,"ECL_95":ecl_95,"Climate_Capital_VaR":ecl_95}
            log_model_run("MonteCarlo",{"company":company_name,"pd_mean":pd_mean,"pd_95":pd_95,"ecl_95":ecl_95})

        # ── REVERSE STRESS TEST ──
        st.divider()
        st.subheader("🔥 Reverse Stress Test")
        run_rev=st.checkbox("Run Reverse Stress Test")
        if run_rev:
            tgt_pd=st.slider("Target PD Threshold",0.05,0.40,0.15)
            breach=None
            for cp in np.linspace(50,500,50):
                cc=TOTAL_EMISSIONS*cp*USD_INR/1e7; cb=cc/revenue_0
                lp=logit(base_pd)+beta_carbon_credit*cb+alpha_dscr*np.clip(1.5-(dscr_trans or 1.5),-4,6)
                if max(base_pd, sigmoid(lp))>=tgt_pd: breach=cp; break
            if breach: st.error(f"⚠️ PD exceeds {tgt_pd:.2%} when carbon price reaches **${breach:.0f}/tCO₂**")
            else: st.success("No breach in tested carbon price range (up to $500/tCO₂)")

        log_model_run("Integrated",{"company":company_name,"sector":sector,"pd_integrated":pd_integrated,
            "ecl_integrated":ecl_integrated,"risk_score":risk_score_int,"capital_signal":capital_signal})

    # ============================================================
    # TAB 6 — PLOTS  (comprehensive, all modules)
    # ============================================================
with plots_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>📈 Visual Analytics Centre</h2>",unsafe_allow_html=True)
    st.caption("All scenario charts, risk analytics, BRSR diagnostics, physical risk maps, and target comparisons in one place.")

    t_ran  = st.session_state.get("transition_ran", False)
    p_ran  = st.session_state.get("physical_ran",   False)
    b_ran  = st.session_state.get("brsr_ran",       False)
    tg_ran = st.session_state.get("targets_ran",    False)

    if not (t_ran or p_ran or b_ran):
        st.info("Run at least one engine to see charts.")
    else:

        # Scenario filter (only if transition ran)
        scen_opts = []
        if t_ran:
            df_tp = st.session_state.get("df_transition")
            if isinstance(df_tp, pd.DataFrame) and not df_tp.empty:
                scen_opts = sorted(df_tp["Scenario"].unique())
        sel_scens = st.multiselect("Filter Scenarios", options=scen_opts, default=scen_opts, key="plots_scen")

        # ── SECTION 1: TRANSITION RISK ──────────────────────────────────
        if t_ran and scen_opts:
            df_tp = st.session_state.get("df_transition")
            if isinstance(df_tp, pd.DataFrame) and not df_tp.empty:
                df_tp_f = df_tp[df_tp["Scenario"].isin(sel_scens)] if sel_scens else df_tp

                st.markdown(f"<h3 style='color:{C['accent2']};margin-top:8px;'>⚡ Transition Risk Analytics</h3>", unsafe_allow_html=True)

                # Row 1: PD + ECL
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure()
                    for scen in df_tp_f["Scenario"].unique():
                        ds = df_tp_f[df_tp_f["Scenario"]==scen]
                        fig.add_trace(go.Scatter(x=ds["Year"], y=ds["PD_Transition"],
                            mode="lines+markers", name=scen[:28],
                            line=dict(color=_scen_color(scen), width=2.5), marker=dict(size=7),
                            hovertemplate="Year: %{x}<br>PD: %{y:.2%}<extra>" + scen[:18] + "</extra>"))
                    fig.update_layout(**_chart_layout("Probability of Default — All Scenarios", 300))
                    fig.update_yaxes(tickformat=".1%")
                    _ax_style(fig)
                    st.plotly_chart(fig, width="stretch")

                with col2:
                    fig = go.Figure()
                    for scen in df_tp_f["Scenario"].unique():
                        ds = df_tp_f[df_tp_f["Scenario"]==scen]
                        fig.add_trace(go.Bar(x=ds["Year"], y=ds["ECL_Transition"],
                            name=scen[:28], marker_color=_scen_color(scen), opacity=0.85,
                            hovertemplate="Year: %{x}<br>ECL: ₹%{y:.1f} Cr<extra>" + scen[:18] + "</extra>"))
                    fig.update_layout(**_chart_layout("Expected Credit Loss (₹ Cr)", 300), barmode="group")
                    _ax_style(fig)
                    st.plotly_chart(fig, width="stretch")

                # Row 2: DSCR + Carbon Burden
                col3, col4 = st.columns(2)
                with col3:
                    fig = go.Figure()
                    for scen in df_tp_f["Scenario"].unique():
                        ds = df_tp_f[df_tp_f["Scenario"]==scen]
                        fig.add_trace(go.Scatter(x=ds["Year"], y=ds["DSCR"],
                            mode="lines+markers", name=scen[:28],
                            line=dict(color=_scen_color(scen), width=2), marker=dict(size=6)))
                    fig.add_hline(y=1.2, line_dash="dash", line_color=C["coral"],
                        annotation_text="1.2× covenant", annotation_font_color=C["coral"])
                    fig.add_hline(y=1.5, line_dash="dot", line_color=C["amber"],
                        annotation_text="1.5× threshold", annotation_font_color=C["amber"])
                    fig.update_layout(**_chart_layout("DSCR Stress Trajectory", 300))
                    _ax_style(fig)
                    st.plotly_chart(fig, width="stretch")

                with col4:
                    fig = go.Figure()
                    for scen in df_tp_f["Scenario"].unique():
                        ds = df_tp_f[df_tp_f["Scenario"]==scen]
                        fig.add_trace(go.Scatter(x=ds["Year"], y=ds["Carbon_Burden"]*100,
                            mode="lines+markers", name=scen[:28],
                            line=dict(color=_scen_color(scen), width=2), marker=dict(size=6),
                            fill="tozeroy", fillcolor=_hex_rgba(_scen_color(scen), 0.13)))
                    fig.update_layout(**_chart_layout("Carbon Burden (% of Revenue)", 300))
                    fig.update_yaxes(ticksuffix="%")
                    _ax_style(fig)
                    st.plotly_chart(fig, width="stretch")

                # Row 3: EBITDA Margin + Stranded Assets + CAPEX Gap (3-panel subplot)
                fig3 = make_subplots(rows=1, cols=3,
                    subplot_titles=("EBITDA Margin (%)", "Stranded Assets (₹ Cr)", "CAPEX Gap / Surplus (₹ Cr)"))
                for scen in df_tp_f["Scenario"].unique():
                    ds = df_tp_f[df_tp_f["Scenario"]==scen]; clr = _scen_color(scen)
                    fig3.add_trace(go.Scatter(x=ds["Year"], y=ds["EBITDA_Margin"]*100, mode="lines+markers",
                        name=scen[:20], line=dict(color=clr, width=2), marker=dict(size=5),
                        showlegend=True), row=1, col=1)
                    fig3.add_trace(go.Scatter(x=ds["Year"], y=ds["Stranded_Assets"], mode="lines+markers",
                        name=scen[:20], line=dict(color=clr, width=2), marker=dict(size=5),
                        showlegend=False), row=1, col=2)
                    fig3.add_trace(go.Scatter(x=ds["Year"], y=ds["CAPEX_Gap"], mode="lines+markers",
                        name=scen[:20], line=dict(color=clr, width=2), marker=dict(size=5),
                        showlegend=False), row=1, col=3)
                fig3.update_layout(**_chart_layout("Financial Stress Indicators by Scenario", 280,
                    legend_override=dict(orientation="h", y=1.12)))
                _ax_style(fig3, rows=1, cols=3)
                fig3.update_yaxes(ticksuffix="%", row=1, col=1)
                st.plotly_chart(fig3, width="stretch")

                # Row 4: Carbon burden vs EBITDA scatter
                fig_sc = go.Figure()
                for scen in df_tp_f["Scenario"].unique():
                    ds = df_tp_f[df_tp_f["Scenario"]==scen]
                    fig_sc.add_trace(go.Scatter(x=ds["Carbon_Burden"]*100, y=ds["EBITDA_Margin"]*100,
                        mode="markers+lines", name=scen[:28],
                        marker=dict(color=_scen_color(scen), size=9,
                            line=dict(color=C["white"], width=0.5)),
                        line=dict(color=_scen_color(scen), width=1, dash="dot"),
                        text=ds["Year"].astype(str), textposition="top center",
                        hovertemplate="Carbon Burden: %{x:.1f}%<br>EBITDA Margin: %{y:.1f}%<extra>" + scen[:18] + "</extra>"))
                fig_sc.update_layout(**_chart_layout("Carbon Burden vs EBITDA Margin — Stress Path", 300))
                fig_sc.update_xaxes(title="Carbon Burden (%)"); fig_sc.update_yaxes(title="EBITDA Margin (%)")
                _ax_style(fig_sc)
                st.plotly_chart(fig_sc, width="stretch")

        # ── SECTION 2: TARGETS COMPARISON ──────────────────────────────
        if tg_ran:
            st.markdown(f"<h3 style='color:{C['amber']};margin-top:8px;'>🎯 Targets — Baseline vs Achievement</h3>", unsafe_allow_html=True)
            df_b2  = st.session_state.get("df_transition")
            df_tg2 = st.session_state.get("df_target")
            if isinstance(df_b2, pd.DataFrame) and isinstance(df_tg2, pd.DataFrame):
                df_pl2 = df_b2.merge(df_tg2, on=["Scenario","Year"], how="inner")
                df_pl2 = df_pl2[df_pl2["Scenario"].isin(sel_scens)] if sel_scens else df_pl2

                tc1, tc2 = st.columns(2)
                with tc1:
                    fig_tp = go.Figure()
                    for scen in df_pl2["Scenario"].unique():
                        ds = df_pl2[df_pl2["Scenario"]==scen]; clr = _scen_color(scen)
                        fig_tp.add_trace(go.Scatter(x=ds["Year"], y=ds["PD_Transition"], mode="lines",
                            name=f"{scen[:18]} Base", line=dict(dash="dash", color=clr, width=1.5)))
                        fig_tp.add_trace(go.Scatter(x=ds["Year"], y=ds["PD_Target"], mode="lines+markers",
                            name=f"{scen[:18]} Target", line=dict(color=clr, width=2.5), marker=dict(size=6)))
                    fig_tp.update_layout(**_chart_layout("PD — Baseline vs Target", 300))
                    fig_tp.update_yaxes(tickformat=".1%")
                    _ax_style(fig_tp)
                    st.plotly_chart(fig_tp, width="stretch")

                with tc2:
                    fig_te = go.Figure()
                    for scen in df_pl2["Scenario"].unique():
                        ds = df_pl2[df_pl2["Scenario"]==scen]; clr = _scen_color(scen)
                        fig_te.add_trace(go.Scatter(x=ds["Year"], y=ds["ECL_Transition"], mode="lines",
                            name=f"{scen[:18]} Base", line=dict(dash="dash", color=clr, width=1.5)))
                        fig_te.add_trace(go.Scatter(x=ds["Year"], y=ds["ECL_Target"], mode="lines+markers",
                            name=f"{scen[:18]} Target", line=dict(color=clr, width=2.5), marker=dict(size=6)))
                    fig_te.update_layout(**_chart_layout("ECL — Baseline vs Target (₹ Cr)", 300))
                    _ax_style(fig_te)
                    st.plotly_chart(fig_te, width="stretch")

                # BRSR target comparison if available
                if b_ran and "brsr_target_overlay" in st.session_state:
                    curr_up   = st.session_state.get("brsr_pd_adj", 0) * 10000
                    tgt_up    = st.session_state["brsr_target_overlay"] * 10000
                    reduction = curr_up - tgt_up
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Bar(name="Current BRSR Uplift", x=["BRSR Governance Signal"], y=[curr_up],
                        marker_color=C["coral"], text=[f"{curr_up:.0f} bps"], textposition="outside"))
                    fig_bt.add_trace(go.Bar(name="Post-Target Uplift", x=["BRSR Governance Signal"], y=[tgt_up],
                        marker_color=C["mint"], text=[f"{tgt_up:.0f} bps"], textposition="outside"))
                    fig_bt.update_layout(**_chart_layout(f"BRSR Governance Signal: {curr_up:.0f} → {tgt_up:.0f} bps (−{reduction:.0f} bps reduction)", 250),
                        barmode="group", yaxis_title="Basis Points")
                    _ax_style(fig_bt)
                    st.plotly_chart(fig_bt, width="stretch")

        # ── SECTION 3: BRSR ANALYTICS ─────────────────────────────────
        if b_ran:
            st.markdown(f"<h3 style='color:{C['mint']};margin-top:8px;'>📘 BRSR Operational Risk Analytics</h3>", unsafe_allow_html=True)
            bs = st.session_state.get("brsr_summary")
            bf = st.session_state.get("brsr_flags")

            if isinstance(bs, pd.DataFrame) and not bs.empty:
                row_b = bs.iloc[0]
                bc1, bc2 = st.columns(2)

                # GHG intensity gauge
                with bc1:
                    ghg_val = float(row_b.get("GHG_Intensity", 0))
                    bench   = SECTOR_GHG_BENCHMARKS.get(sector, SECTOR_GHG_BENCHMARKS["Manufacturing"])
                    fig_gb = go.Figure(go.Bar(
                        x=["Your GHG", "Sector P25", "Sector P50", "Sector P75"],
                        y=[ghg_val, bench["p25"], bench["p50"], bench["p75"]],
                        marker_color=[
                            C["mint"] if ghg_val <= bench["p50"] else C["amber"] if ghg_val <= bench["p75"] else C["coral"],
                            C["mint"], C["amber"], C["coral"]
                        ],
                        text=[f"{v:.2f}" for v in [ghg_val, bench["p25"], bench["p50"], bench["p75"]]],
                        textposition="outside",
                    ))
                    fig_gb.update_layout(**_chart_layout("GHG Intensity vs Sector Benchmarks (tCO₂/₹Cr)", 280))
                    fig_gb.update_yaxes(title="tCO₂e / ₹Cr")
                    _ax_style(fig_gb)
                    st.plotly_chart(fig_gb, width="stretch")

                # BRSR risk overlay breakdown
                with bc2:
                    if isinstance(bf, pd.DataFrame) and "Flag" in bf.columns and len(bf) > 0:
                        flags_list = bf["Flag"].tolist()
                        flag_bps = {f: BRSR_PD_SPREAD.get(f, 0)*10000 for f in flags_list}
                        fig_fb = go.Figure(go.Bar(
                            x=list(flag_bps.values()), y=list(flag_bps.keys()),
                            orientation="h",
                            marker_color=[C["coral"] if v>=25 else C["amber"] if v>=15 else C["mint"] for v in flag_bps.values()],
                            text=[f"+{v:.0f} bps" for v in flag_bps.values()], textposition="outside",
                        ))
                        fig_fb.update_layout(
                            **_chart_layout(f"BRSR Flags → Risk Overlay (Total: +{sum(flag_bps.values()):.0f} bps)", 280,
                                margin_override=dict(l=230, r=80, t=50, b=20)),
                        )
                        _ax_style(fig_fb)
                        st.plotly_chart(fig_fb, width="stretch")
                    else:
                        st.success("✅ No BRSR flags — zero risk overlay from operational climate risk")

                # Regulatory readiness radar
                rdness_cols = {"Renewable ≥20%": float(row_b.get("Renewable_%",0))>=20,
                               "Coverage ≥50%":  float(row_b.get("Target_Coverage_%",0))>=50,
                               "Haz Waste ≤30%": float(row_b.get("Hazardous_Waste_%",100))<=30,
                               "Water Risk OK":   row_b.get("Water_Stress_Level","High")!="High"}
                rdness_scores = [100 if v else 0 for v in rdness_cols.values()]
                rcats = list(rdness_cols.keys())
                fig_rad = go.Figure()
                fig_rad.add_trace(go.Scatterpolar(
                    r=rdness_scores + [rdness_scores[0]], theta=rcats + [rcats[0]],
                    fill="toself", fillcolor="rgba(6,182,212,0.18)",
                    line=dict(color=C["accent2"], width=2), name="Current"))
                fig_rad.add_trace(go.Scatterpolar(
                    r=[100]*(len(rcats)+1), theta=rcats+[rcats[0]],
                    line=dict(color=C["mint"], width=1, dash="dot"), name="Target"))
                fig_rad.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,100]), bgcolor=C["bg_dark"]),
                    **_chart_layout("SEBI BRSR Core Readiness", 320),
                    showlegend=True,
                )
                st.plotly_chart(fig_rad, width="stretch")

        # ── SECTION 4: PHYSICAL RISK ───────────────────────────────────
        if p_ran:
            st.markdown(f"<h3 style='color:{C['coral']};margin-top:8px;'>🌍 Physical Risk Analytics</h3>", unsafe_allow_html=True)
            phys = st.session_state.get("phys_assets")
            df_pp = st.session_state.get("df_physical_projection")

            if isinstance(phys, pd.DataFrame) and not phys.empty:
                pc1, pc2 = st.columns(2)

                # Asset revenue loss bar
                with pc1:
                    fig_rl = go.Figure()
                    for haz, col, clr in [("H_flood","Flood",C["accent3"]),("H_heat","Heat",C["amber"]),("H_cyclone","Cyclone",C["coral"])]:
                        if haz in phys.columns:
                            # Prefer pre-computed normalized revenue-loss components from Physical module.
                            comp_col = {"H_flood":"rl_flood","H_heat":"rl_heat","H_cyclone":"rl_cyclone"}.get(haz)
                            rl = phys[comp_col] if comp_col in phys.columns else phys["revenue_loss"] * phys[haz] / phys["D_total"].replace(0,1)
                            fig_rl.add_trace(go.Bar(x=phys["asset_id"], y=rl.fillna(0), name=col, marker_color=clr))
                    fig_rl.update_layout(**_chart_layout("Revenue Loss by Asset & Hazard (₹ Cr)", 300), barmode="stack")
                    _ax_style(fig_rl)
                    st.plotly_chart(fig_rl, width="stretch")

                # Asset damage heatmap
                with pc2:
                    hm_cols = {"H_flood":"Flood","H_heat":"Heat","H_cyclone":"Cyclone"}
                    hm = phys.set_index("asset_id")[[c for c in hm_cols if c in phys.columns]].rename(columns=hm_cols)*100
                    if not hm.empty:
                        fig_hm = go.Figure(go.Heatmap(
                            z=hm.values.T, x=hm.index.tolist(), y=hm.columns.tolist(),
                            colorscale=[[0,C["mint"]],[0.35,C["amber"]],[0.65,"#F97316"],[1.0,C["coral"]]],
                            text=[[f"{v:.0f}" for v in row] for row in hm.values.T],
                            texttemplate="%{text}", colorbar=dict(title="Risk Score"),
                            zmin=0, zmax=100,
                        ))
                        fig_hm.update_layout(**_chart_layout("Asset Vulnerability Heatmap (0–100)", 280))
                        _ax_style(fig_hm)
                        st.plotly_chart(fig_hm, width="stretch")

            # NGFS scenario projections fan chart
            if isinstance(df_pp, pd.DataFrame) and not df_pp.empty:
                scens_p = sorted(df_pp["Scenario"].unique())
                fig_fan = make_subplots(
                    rows=1, cols=len(scens_p),
                    subplot_titles=[s[:22] for s in scens_p],
                    shared_yaxes=True,
                )
                for ci, scen in enumerate(scens_p, 1):
                    ds = df_pp[df_pp["Scenario"]==scen].sort_values("Year")
                    clr = _scen_color(scen)
                    # Uncertainty band
                    fig_fan.add_trace(go.Scatter(
                        x=ds["Year"].tolist()+ds["Year"].tolist()[::-1],
                        y=ds["Revenue_Loss_P90_Cr"].tolist()+ds["Revenue_Loss_P10_Cr"].tolist()[::-1],
                        fill="toself", fillcolor=_hex_rgba(clr, 0.16), line=dict(color="rgba(0,0,0,0)"),
                        name="P10–P90", showlegend=(ci==1),
                    ), row=1, col=ci)
                    fig_fan.add_trace(go.Scatter(x=ds["Year"], y=ds["Revenue_Loss_P50_Cr"],
                        mode="lines+markers", line=dict(color=clr, width=2.5),
                        marker=dict(size=6), name=f"{scen[:18]} P50", showlegend=(ci==1),
                    ), row=1, col=ci)
                fig_fan.update_layout(**_chart_layout("Physical Revenue Loss — NGFS Scenarios · P10/P50/P90 Bands", 320,
                    legend_override=dict(orientation="h", y=1.1)))
                _ax_style(fig_fan, rows=1, cols=len(scens_p))
                st.plotly_chart(fig_fan, width="stretch")

                # PD evolution across scenarios
                fig_pd_p = go.Figure()
                for scen in scens_p:
                    ds = df_pp[df_pp["Scenario"]==scen].sort_values("Year")
                    fig_pd_p.add_trace(go.Scatter(x=ds["Year"], y=ds["PD_Physical"],
                        mode="lines+markers", name=scen[:28],
                        line=dict(color=_scen_color(scen), width=2.5), marker=dict(size=7)))
                fig_pd_p.update_layout(**_chart_layout("Physical PD — NGFS Scenario Projections", 280))
                fig_pd_p.update_yaxes(tickformat=".2%")
                _ax_style(fig_pd_p)
                st.plotly_chart(fig_pd_p, width="stretch")

            # Folium map
            if isinstance(phys, pd.DataFrame) and not phys.empty:
                hchoice = st.selectbox("Hazard map layer", ["Integrated","Flood","Heat","Cyclone"], key="plots_haz")
                m = folium.Map(location=[phys["latitude"].mean(), phys["longitude"].mean()],
                    zoom_start=5, tiles="cartodbdark_matter")
                for _, r in phys.iterrows():
                    v = (r["H_flood"] if hchoice=="Flood" else r["H_heat"] if hchoice=="Heat"
                         else r["H_cyclone"] if hchoice=="Cyclone" else r["D_total"])
                    clr = "green" if v<0.2 else "orange" if v<0.4 else "red"
                    folium.CircleMarker(
                        location=[r["latitude"], r["longitude"]], radius=9,
                        color=clr, fill=True, fill_opacity=0.78,
                        popup=f"<b>{r['asset_id']}</b><br>Risk: {v:.2f}<br>Rev Loss: ₹{r['revenue_loss']:.1f}Cr"
                    ).add_to(m)
                st_folium(m, width=1200, height=400)

        # ── SECTION 5: INTEGRATED RISK SUMMARY CHART ──────────────────
        if st.session_state.get("integrated_ran", False):
            st.markdown(f"<h3 style='color:{C['purple']};margin-top:8px;'>🧩 Integrated Risk Summary</h3>", unsafe_allow_html=True)
            df_int = st.session_state.get("df_integrated_summary")
            if isinstance(df_int, pd.DataFrame) and not df_int.empty:
                # Extract numeric values for waterfall/bar display
                _rows = df_int[df_int["Metric"].str.contains("PD|ECL|DSCR|Score", na=False)]
                if not _rows.empty:
                    labels = _rows["Metric"].tolist()
                    vals   = pd.to_numeric(_rows["Value"], errors="coerce").fillna(0).tolist()
                    fig_int = go.Figure(go.Bar(
                        x=[l[:35] for l in labels], y=vals,
                        marker_color=[C["accent2"],C["amber"],C["coral"],C["mint"],C["purple"]][:len(labels)],
                        text=[f"{v:.3f}" if isinstance(v,float) else str(v) for v in vals],
                        textposition="outside",
                    ))
                    fig_int.update_layout(**_chart_layout("Integrated Risk Metrics (normalised values)", 280))
                    _ax_style(fig_int)
                    st.plotly_chart(fig_int, width="stretch")

            # MC distribution if available
            mc = st.session_state.get("mc_results")
            if mc and "ECL_95" in mc:
                st.markdown(f"**Monte Carlo summary:** Mean PD={mc.get('Mean_PD',0):.2%} · PD 95th={mc.get('PD_95',0):.2%} · Climate VaR (ECL 95th)=₹{mc.get('ECL_95',0):.1f} Cr")

    # ============================================================
    # TAB 6 — BRSR CORE  (BRSR-01)
    # ============================================================
with brsr_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>📘 BRSR Core — Climate Risk Intelligence</h2>",unsafe_allow_html=True)
    st.caption("SEBI BRSR Core diagnostics with financial quantification · GHG benchmarking · PD linkage · Regulatory readiness")

    if not st.session_state.get("enable_brsr", False):
        st.info("Enable BRSR Diagnostics in the sidebar.")
    else:

        st.subheader("🔢 Operational Climate Inputs")
        bc1,bc2,bc3=st.columns(3)
        with bc1:
            st.markdown(f"<div style='color:{C['accent2']};font-weight:600;margin-bottom:8px;'>⚡ Energy</div>",unsafe_allow_html=True)
            total_energy_kwh    =st.number_input("Total Energy (kWh)",value=1_000_000_000,step=10_000_000,key="brsr_en")
            renewable_share_pct =st.slider("Renewable Share (%)",0,100,15,key="brsr_rs")
            has_verified_data   =st.checkbox("Third-Party Verified Data",value=False,key="brsr_vd")
        with bc2:
            st.markdown(f"<div style='color:{C['amber']};font-weight:600;margin-bottom:8px;'>💧 Water</div>",unsafe_allow_html=True)
            total_water_m3      =st.number_input("Water Withdrawal (m³)",value=50_000_000,step=1_000_000,key="brsr_wt")
            water_stress_region =st.selectbox("Water Stress Region",["Low","Medium","High"],key="brsr_ws")
            recycled_water_pct  =st.slider("Recycled Water (%)",0,100,10,key="brsr_rw")
        with bc3:
            st.markdown(f"<div style='color:{C['coral']};font-weight:600;margin-bottom:8px;'>♻️ Waste & Targets</div>",unsafe_allow_html=True)
            total_waste_mt      =st.number_input("Waste Generated (MT)",value=100_000,step=1_000,key="brsr_wa")
            hazardous_waste_pct =st.slider("Hazardous Waste Share (%)",0,100,20,key="brsr_hw")
            target_coverage_pct =st.slider("Target Coverage (%)",0,100,50,key="brsr_tc")

        st.markdown(f"<div style='color:{C['mint']};font-weight:600;margin:8px 0 4px;'>📋 Governance</div>",unsafe_allow_html=True)
        bg1,bg2,bg3,bg4=st.columns(4)
        has_scope3          =bg1.checkbox("Scope 3 Disclosed",value=True,key="brsr_s3")
        has_board_oversight =bg2.checkbox("Board Climate Oversight",value=False,key="brsr_bo")
        has_net_zero_target =bg3.checkbox("Long-term Climate Target",value=False,key="brsr_nz")
        has_cbam            =bg4.checkbox("EU/UK Export Exposure",value=False,key="brsr_cb")

        # Locked public demo: BRSR inputs are fixed to the fictional company case.
        # Widgets above are displayed for product feel, but public demo output remains standardized.
        total_energy_kwh = 1_000_000_000
        renewable_share_pct = 15
        has_verified_data = False
        total_water_m3 = 50_000_000
        water_stress_region = "High"
        recycled_water_pct = 10
        total_waste_mt = 100_000
        hazardous_waste_pct = 20
        target_coverage_pct = 50
        has_scope3 = True
        has_board_oversight = False
        has_net_zero_target = False
        has_cbam = False
        st.caption("🔒 Public demo uses locked BRSR inputs for the fictional company. Custom inputs are available in pilot access.")

        with st.expander("📈 3-Year Emissions Trend (optional)"):
            te1,te2,te3=st.columns(3)
            s1_y0=te1.number_input("Scope 1 (2yr ago)",value=float(scope1*1.08),key="brsr_s1y0")
            s1_y1=te2.number_input("Scope 1 (1yr ago)",value=float(scope1*1.04),key="brsr_s1y1")
            s1_y2=te3.number_input("Scope 1 (current)",value=float(scope1),key="brsr_s1y2")
            r_y0=te1.number_input("Revenue (2yr ago, ₹Cr)",value=float(revenue_0*0.92),key="brsr_r0")
            r_y1=te2.number_input("Revenue (1yr ago, ₹Cr)",value=float(revenue_0*0.96),key="brsr_r1")
            r_y2=te3.number_input("Revenue (current, ₹Cr)",value=float(revenue_0),key="brsr_r2")

        run_brsr=st.button("▶ Run BRSR Diagnostics",type="primary",key="run_brsr_btn")
        if not run_brsr:
            if not render_cached_brsr_results():
                st.info("Click Run BRSR Diagnostics to proceed.")
        else:

            # ── COMPUTE ──
            ghg_bench=SECTOR_GHG_BENCHMARKS.get(sector,SECTOR_GHG_BENCHMARKS["Manufacturing"])
            en_bench =SECTOR_ENERGY_BENCHMARKS.get(sector,SECTOR_ENERGY_BENCHMARKS["Manufacturing"])

            ghg_int = TOTAL_EMISSIONS/max(revenue_0,1e-6)
            s1_int  = scope1/max(revenue_0,1e-6); s2_int=scope2/max(revenue_0,1e-6); s3_int=scope3/max(revenue_0,1e-6)
            en_int  = total_energy_kwh/max(revenue_0*1e6,1)
            wa_int  = total_water_m3/max(revenue_0*1e6,1)

            def pctile(v,b):
                if v<=b["p25"]: return "Top 25% 🟢"
                if v<=b["p50"]: return "25–50th 🟡"
                if v<=b["p75"]: return "50–75th 🟠"
                return "Bottom 25% 🔴"
            ghg_rank=pctile(ghg_int,ghg_bench)

            flags=[]
            if ghg_int>ghg_bench["p75"]:         flags.append("High GHG intensity (>P75)")
            if renewable_share_pct<20:           flags.append("Low renewable energy (<20%)")
            if en_int>en_bench["p75"]:           flags.append("High energy intensity (>P75)")
            if water_stress_region=="High":      flags.append("High water stress + high withdrawal")
            if target_coverage_pct<50:          flags.append("Low target coverage (<50%)")
            if hazardous_waste_pct>30:          flags.append("High hazardous waste (>30%)")
            if not has_verified_data:           flags.append("No verified emissions data")
            if not has_scope3:                  flags.append("Missing Scope 3 disclosure")

            pd_adj=float(np.clip(sum(BRSR_PD_SPREAD.get(f,0) for f in flags),0,0.015))

            # Water stranded cost
            esc_r={"Low":0.03,"Medium":0.08,"High":0.18}[water_stress_region]
            w_cum_cost=sum(total_water_m3*42*((1+esc_r)**yr-(1+esc_r)**(yr-1))/1e7 for yr in range(1,6))
            # Energy transition risk
            fossil_kwh=total_energy_kwh*(100-renewable_share_pct)/100
            energy_tr_risk=fossil_kwh*2.0/1e7

            # Regulatory readiness (8-point)
            rdness_items={
                "Renewable ≥20%":1.0 if renewable_share_pct>=20 else 0.5 if renewable_share_pct>=10 else 0.0,
                "Target Coverage ≥50%":1.0 if target_coverage_pct>=50 else 0.5 if target_coverage_pct>=25 else 0.0,
                "Hazardous Waste ≤30%":1.0 if hazardous_waste_pct<=30 else 0.5 if hazardous_waste_pct<=40 else 0.0,
                "Water Risk Mgmt":1.0 if water_stress_region=="Low" else 0.5 if water_stress_region=="Medium" else 0.0,
                "Scope 3 Disclosed":1.0 if has_scope3 else 0.0,
                "Verified Data":1.0 if has_verified_data else 0.0,
                "Board Oversight":1.0 if has_board_oversight else 0.0,
                "Long-term Climate Target":1.0 if has_net_zero_target else 0.0,
            }
            readiness_score=sum(rdness_items.values())/len(rdness_items)*100

            risk_score_brsr=0
            if renewable_share_pct<10: risk_score_brsr+=20
            elif renewable_share_pct<20: risk_score_brsr+=12
            if hazardous_waste_pct>40: risk_score_brsr+=15
            elif hazardous_waste_pct>30: risk_score_brsr+=8
            if target_coverage_pct<30: risk_score_brsr+=18
            elif target_coverage_pct<50: risk_score_brsr+=10
            if water_stress_region=="High": risk_score_brsr+=18
            elif water_stress_region=="Medium": risk_score_brsr+=8
            if ghg_int>ghg_bench["p75"]: risk_score_brsr+=15
            elif ghg_int>ghg_bench["p50"]: risk_score_brsr+=7
            if not has_board_oversight: risk_score_brsr+=8
            if not has_net_zero_target: risk_score_brsr+=6
            risk_score_brsr=min(100,risk_score_brsr)

            sev_brsr="🟢 Low" if risk_score_brsr<30 else "🟡 Moderate" if risk_score_brsr<60 else "🔴 High"
            sev_color_brsr=C["mint"] if risk_score_brsr<30 else C["amber"] if risk_score_brsr<60 else C["coral"]

            brsr_summary=pd.DataFrame([{"GHG_Intensity":round(ghg_int,3),"S1_Intensity":round(s1_int,3),
                "S2_Intensity":round(s2_int,3),"S3_Intensity":round(s3_int,3),"Energy_Intensity":round(en_int,3),
                "Renewable_%":renewable_share_pct,"Water_Intensity":round(wa_int,3),"Recycled_Water_%":recycled_water_pct,
                "Hazardous_Waste_%":hazardous_waste_pct,"Target_Coverage_%":target_coverage_pct,
                "BRSR_PD_Adj_bps":round(pd_adj*10000,1),"Water_Cost_Risk_Cr":round(w_cum_cost,2),
                "Energy_TR_Risk_Cr":round(energy_tr_risk,2),"Readiness_Score":round(readiness_score,1),
                "Overall_Risk_Score":risk_score_brsr}])
            st.session_state["brsr_summary"]=brsr_summary
            st.session_state["brsr_flags"]=pd.DataFrame({"Flag":flags})
            st.session_state["brsr_pd_adj"]=pd_adj
            st.session_state["brsr_ran"]=True

            # ── DISPLAY ──
            st.markdown(f"""
            <div style="background:{C['card']};border:1px solid {sev_color_brsr};border-radius:10px;padding:14px 18px;margin-bottom:12px;display:flex;align-items:center;gap:16px;">
              <div style="font-size:30px;">{sev_brsr.split()[0]}</div>
              <div>
                <div style="font-size:16px;font-weight:700;color:{sev_color_brsr};">{sev_brsr} Operational Climate Risk · Score: {risk_score_brsr}/100</div>
                <div style="font-size:13px;color:{C['slate']};margin-top:2px;">{len(flags)} flags · risk overlay: +{pd_adj*10000:.0f}bps · Readiness: {readiness_score:.0f}% · Water cost risk (5yr): ₹{w_cum_cost:.1f}Cr</div>
              </div>
            </div>""",unsafe_allow_html=True)

            bm1,bm2,bm3,bm4,bm5=st.columns(5)
            bm1.metric("Risk Score",f"{risk_score_brsr}/100")
            bm2.metric("BRSR Readiness",f"{readiness_score:.0f}%")
            bm3.metric("Risk Overlay",f"+{pd_adj*10000:.0f} bps")
            bm4.metric("Water Cost Risk (5yr)",f"₹{w_cum_cost:.1f} Cr")
            bm5.metric("Energy Transition Risk",f"₹{energy_tr_risk:.1f} Cr")

            # Chart 1: GHG benchmark
            st.subheader("📊 GHG Intensity — Sector Benchmark")
            fig_ghg=make_subplots(rows=1,cols=2,subplot_titles=("vs Sector Percentiles","Scope Breakdown"))
            bench_vals=[ghg_bench["p25"],ghg_bench["p50"],ghg_bench["p75"],ghg_int]
            bench_clrs=[C["mint"],C["amber"],"#F97316",C["accent2"] if ghg_int<=ghg_bench["p75"] else C["coral"]]
            fig_ghg.add_trace(go.Bar(x=["P25","P50","P75","This Company"],y=bench_vals,marker_color=bench_clrs,
                text=[f"{v:.2f}" for v in bench_vals],textposition="outside"),row=1,col=1)
            fig_ghg.add_trace(go.Bar(x=["Scope 1","Scope 2","Scope 3"],y=[s1_int,s2_int,s3_int],
                marker_color=[C["coral"],C["amber"],C["accent3"]],text=[f"{v:.2f}" for v in [s1_int,s2_int,s3_int]],textposition="outside"),row=1,col=2)
            fig_ghg.update_layout(**_chart_layout("",300)); _ax_style(fig_ghg,rows=1,cols=2)
            fig_ghg.update_yaxes(title="tCO₂e/₹Cr",row=1,col=1); fig_ghg.update_yaxes(title="tCO₂e/₹Cr",row=1,col=2)
            st.plotly_chart(fig_ghg,width="stretch")
            st.caption(f"Sector ({sector}): P25={ghg_bench['p25']}, P50={ghg_bench['p50']}, P75={ghg_bench['p75']} tCO₂e/₹Cr · Your intensity: **{ghg_int:.2f}** — {ghg_rank}")

            # Chart 2: risk overlay tornado
            if flags:
                st.subheader("🔗 BRSR Flags → Risk Overlay")
                flag_bps={f:BRSR_PD_SPREAD.get(f,0)*10000 for f in flags}
                fig_tor=go.Figure(go.Bar(x=list(flag_bps.values()),y=list(flag_bps.keys()),orientation="h",
                    marker_color=[C["coral"] if v>=25 else C["amber"] if v>=15 else C["mint"] for v in flag_bps.values()],
                    text=[f"+{v:.0f} bps" for v in flag_bps.values()],textposition="outside"))
                fig_tor.update_layout(**_chart_layout(f"Risk Overlay by BRSR Flag (Total: +{pd_adj*10000:.0f}bps)",max(220,len(flags)*45+80),
                    margin_override=dict(l=250,r=80,t=50,b=20)))
                fig_tor.update_xaxes(title="Basis Point Uplift")
                _ax_style(fig_tor); st.plotly_chart(fig_tor,width="stretch")
            else:
                st.success("✅ No BRSR flags — no risk overlay from operational climate risk")

            # Chart 3: Readiness radar
            st.subheader("🎯 SEBI BRSR Core Regulatory Readiness")
            rcol1,rcol2=st.columns([2,1])
            with rcol1:
                rcats=list(rdness_items.keys()); rvals=[rdness_items[c]*100 for c in rcats]
                fig_rad=go.Figure()
                fig_rad.add_trace(go.Scatterpolar(r=rvals+[rvals[0]],theta=rcats+[rcats[0]],fill="toself",
                    fillcolor="rgba(6,182,212,0.15)",line=dict(color=C["accent2"],width=2),name="Readiness"))
                fig_rad.add_trace(go.Scatterpolar(r=[100]*(len(rcats)+1),theta=rcats+[rcats[0]],
                    line=dict(color=C["mint"],width=1,dash="dot"),name="Full Compliance"))
                fig_rad.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100]),bgcolor=C["bg_dark"]),
                    **_chart_layout("",360),showlegend=True)
                st.plotly_chart(fig_rad,width="stretch")
            with rcol2:
                st.dataframe(pd.DataFrame([{"Item":k,"Status":"✅" if v==1 else "⚠️" if v==0.5 else "❌"} for k,v in rdness_items.items()])
                    .style.set_properties(**{"background-color":C["bg_dark"],"color":C["text"]}),
                    width="stretch",height=320,hide_index=True)

            # Chart 4: Forward financial risk
            st.subheader("💰 5-Year Forward Operational Climate Cost")
            yr5=list(range(1,6))
            wc_ann=[total_water_m3*42*((1+esc_r)**y-(1+esc_r)**(y-1))/1e7 for y in yr5]
            ec_ann=[fossil_kwh*(0.40*y)/1e7 for y in yr5]
            fig_fwd=go.Figure()
            fig_fwd.add_trace(go.Bar(x=yr5,y=wc_ann,name="Water Cost Escalation",marker_color=C["accent3"]))
            fig_fwd.add_trace(go.Bar(x=yr5,y=ec_ann,name="Energy Carbon Surcharge",marker_color=C["amber"]))
            fig_fwd.update_layout(**_chart_layout("Projected Annual Operational Climate Cost (₹ Cr)",280),barmode="stack")
            _ax_style(fig_fwd); st.plotly_chart(fig_fwd,width="stretch")

            # Chart 5: Emissions trend
            if s1_y0>0 and r_y0>0:
                st.subheader("📈 Scope 1 Intensity Trend")
                ti=[s1_y0/r_y0,s1_y1/r_y1,s1_y2/r_y2]
                cagr=(ti[2]/ti[0])**0.5-1; dir_t="↓ Improving" if cagr<-0.02 else "↑ Worsening" if cagr>0.02 else "→ Stable"
                dir_clr=C["mint"] if cagr<-0.02 else C["coral"] if cagr>0.02 else C["amber"]
                fig_tr=go.Figure()
                fig_tr.add_trace(go.Scatter(x=["2 Years Ago","1 Year Ago","Current"],y=ti,mode="lines+markers",
                    line=dict(color=C["accent2"],width=3),marker=dict(size=10,color=C["accent"]),name="Scope 1 Intensity"))
                fig_tr.add_hline(y=ghg_bench["p50"],line_dash="dash",line_color=C["amber"],annotation_text=f"Sector P50 ({ghg_bench['p50']})",annotation_font_color=C["amber"])
                fig_tr.update_layout(**_chart_layout(f"Scope 1 GHG Intensity Trend — {dir_t} (CAGR: {cagr*100:.1f}%/yr)",260))
                _ax_style(fig_tr); st.plotly_chart(fig_tr,width="stretch")

            # Compliance table
            st.subheader("📋 BRSR Compliance Status")
            crows=[
                {"Indicator":"GHG Intensity vs Sector","Value":f"{ghg_int:.2f} tCO₂/₹Cr","Threshold":f"P50={ghg_bench['p50']}","Status":"🟢" if ghg_int<=ghg_bench["p50"] else "🟡" if ghg_int<=ghg_bench["p75"] else "🔴","Percentile":ghg_rank},
                {"Indicator":"Renewable Energy","Value":f"{renewable_share_pct}%","Threshold":"≥20%","Status":"🟢" if renewable_share_pct>=20 else "🟡" if renewable_share_pct>=10 else "🔴","Percentile":"Compliant" if renewable_share_pct>=20 else "Below"},
                {"Indicator":"Target Coverage","Value":f"{target_coverage_pct}%","Threshold":"≥50%","Status":"🟢" if target_coverage_pct>=50 else "🟡" if target_coverage_pct>=25 else "🔴","Percentile":"Compliant" if target_coverage_pct>=50 else "Below"},
                {"Indicator":"Hazardous Waste","Value":f"{hazardous_waste_pct}%","Threshold":"≤30%","Status":"🟢" if hazardous_waste_pct<=30 else "🔴","Percentile":"Compliant" if hazardous_waste_pct<=30 else "Exceeds"},
                {"Indicator":"Water Stress","Value":water_stress_region,"Threshold":"Low/Medium","Status":"🟢" if water_stress_region!="High" else "🔴","Percentile":"No elevated risk" if water_stress_region!="High" else "High risk"},
                {"Indicator":"Regulatory Readiness","Value":f"{readiness_score:.0f}%","Threshold":"≥75%","Status":"🟢" if readiness_score>=75 else "🟡" if readiness_score>=50 else "🔴","Percentile":"Ready" if readiness_score>=75 else "Needs work"},
            ]
            st.dataframe(pd.DataFrame(crows).style.set_properties(**{"background-color":C["bg_dark"],"color":C["text"]}),
                width="stretch",hide_index=True)

            # Integration link to transition engine
            if st.session_state.get("transition_ran",False):
                df_tl=st.session_state.get("df_transition")
                if isinstance(df_tl,pd.DataFrame) and not df_tl.empty:
                    st.subheader("🔗 Integration: BRSR Uplift on Transition PD")
                    cb_pk=df_tl["Carbon_Burden"].max(); pd_pk=df_tl["PD_Transition"].max()
                    ic1,ic2,ic3=st.columns(3)
                    ic1.metric("Base Transition PD",f"{pd_pk:.2%}")
                    ic2.metric("BRSR Governance Signal",f"+{pd_adj*10000:.0f} bps")
                    ic3.metric("Total PD (Transition × BRSR factor)",f"{min(pd_pk*brsr_governance_multiplier(pd_adj),PD_CAP):.2%}",delta=f"+{pd_adj*10000:.0f}bps",delta_color="inverse")
                    st.caption("BRSR overlay is additive operational climate risk premium. Captures risks outside carbon-price transmission chain: water scarcity, energy cost escalation, governance deficiencies, regulatory non-compliance penalties.")

            st.success("✅ BRSR Enhanced Diagnostics v1.2 completed")
            log_model_run("BRSR",{"company":company_name,"risk_score":risk_score_brsr,"pd_adj_bps":pd_adj*10000,"readiness":readiness_score})

        # ============================================================
        # TAB 7 — AI NARRATIVE
        # ============================================================
with ai_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🤖 AI Business & Climate Intelligence Advisor</h2>", unsafe_allow_html=True)
    st.caption("Seven specialist advisors · Exact problems · Exact solutions · Implementation steps · Business impact")

    t_r = st.session_state.get("transition_ran", False)
    p_r = st.session_state.get("physical_ran",   False)
    b_r = st.session_state.get("brsr_ran",       False)
    i_r = st.session_state.get("integrated_ran", False)

    if not (t_r or p_r or b_r):
        st.info("Run at least one module (Transition Risk, Physical Risk, or BRSR) to unlock the AI advisors.")
    else:

        # ── Build compact AI payload ────────────────────────────────────
        # Groq on-demand has a strict tokens-per-minute limit. Keep payloads
        # compact and advisor-specific so AI cannot fail with 413 request-too-large.
        AI_MAX_PROMPT_CHARS = 5200
        AI_MAX_CONTEXT_CHARS = 900
        AI_MAX_OUTPUT_TOKENS = 650

        def _safe_float(v, default=0.0):
            try:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return default
                return float(v)
            except Exception:
                return default

        def _safe_text(s, max_chars=AI_MAX_CONTEXT_CHARS):
            s = "" if s is None else str(s)
            return s[:max_chars]

        def _compact_json(obj, max_chars=3600):
            txt = json.dumps(obj, ensure_ascii=False, default=str, separators=(",", ":"))
            return txt[:max_chars]

        def _limit_prompt(text, max_chars=AI_MAX_PROMPT_CHARS):
            text = "" if text is None else str(text)
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "\n\n[Input trimmed to stay within Groq token limits.]"

        def _build_compact_payload():
            """Small, decision-ready payload. No full DataFrames or year-by-year dumps."""
            payload = {
                "company": company_name,
                "sector": sector,
                "reporting_year": REPORTING_YEAR,
                "financial_base": {
                    "revenue_cr": round(_safe_float(revenue_0), 2),
                    "ebitda_margin_pct": round(_safe_float(ebitda_margin_0) * 100, 2),
                    "interest_cr": round(_safe_float(interest_payment), 2),
                    "ead_cr": round(_safe_float(exposure_at_default), 2),
                    "emissions_tco2e": round(_safe_float(TOTAL_EMISSIONS), 0),
                    "base_pd_pct": round(_safe_float(base_pd) * 100, 3),
                    "base_lgd_pct": round(_safe_float(LGD_0) * 100, 2),
                },
            }

            if t_r:
                df_ai = st.session_state.get("df_transition")
                if isinstance(df_ai, pd.DataFrame) and not df_ai.empty:
                    worst_pd_idx = df_ai["PD_Transition"].idxmax()
                    worst_dscr_idx = df_ai["DSCR"].idxmin()
                    worst_pd = df_ai.loc[worst_pd_idx]
                    worst_dscr = df_ai.loc[worst_dscr_idx]
                    scenario_rows = []
                    for scen, grp in df_ai.groupby("Scenario"):
                        scenario_rows.append({
                            "scenario": scen,
                            "peak_pd_pct": round(_safe_float(grp["PD_Transition"].max()) * 100, 2),
                            "peak_ecl_cr": round(_safe_float(grp["ECL_Transition"].max()), 2),
                            "min_dscr": round(_safe_float(grp["DSCR"].min()), 2),
                            "max_carbon_burden_pct": round(_safe_float(grp["Carbon_Burden"].max()) * 100, 2),
                            "max_stranded_assets_cr": round(_safe_float(grp["Stranded_Assets"].max()), 2),
                            "max_capex_gap_cr": round(_safe_float(grp["CAPEX_Gap"].max()), 2),
                        })
                    payload["transition"] = {
                        "summary_by_scenario": scenario_rows[:5],
                        "worst_pd": {
                            "scenario": worst_pd.get("Scenario"),
                            "year": int(worst_pd.get("Year")),
                            "pd_pct": round(_safe_float(worst_pd.get("PD_Transition")) * 100, 2),
                            "ecl_cr": round(_safe_float(worst_pd.get("ECL_Transition")), 2),
                            "dscr": round(_safe_float(worst_pd.get("DSCR")), 2),
                        },
                        "worst_dscr": {
                            "scenario": worst_dscr.get("Scenario"),
                            "year": int(worst_dscr.get("Year")),
                            "dscr": round(_safe_float(worst_dscr.get("DSCR")), 2),
                        },
                    }

            if p_r:
                ps = st.session_state.get("phys_summary", {}) or {}
                pa = st.session_state.get("phys_assets")
                top_assets = []
                if isinstance(pa, pd.DataFrame) and not pa.empty and "D_total" in pa.columns:
                    keep_cols = [c for c in ["asset_id", "asset_type", "D_total", "revenue_loss", "downtime_days"] if c in pa.columns]
                    top_assets = pa.nlargest(min(3, len(pa)), "D_total")[keep_cols].round(3).to_dict("records")
                payload["physical"] = {
                    "revenue_loss_cr": round(_safe_float(ps.get("Total Revenue Loss (₹ Cr)", 0)), 2),
                    "ebitda_loss_cr": round(_safe_float(ps.get("EBITDA Loss (₹ Cr)", 0)), 2),
                    "post_risk_dscr": round(_safe_float(ps.get("Post-Risk DSCR", 0)), 2),
                    "physical_pd_pct": round(_safe_float(ps.get("Physical Risk PD", 0)) * 100, 2),
                    "delta_ecl_cr": round(_safe_float(ps.get("ΔECL (₹ Cr)", 0)), 2),
                    "top_risk_assets": top_assets,
                }

            if b_r:
                bs = st.session_state.get("brsr_summary")
                bf = st.session_state.get("brsr_flags")
                row = bs.iloc[0].to_dict() if isinstance(bs, pd.DataFrame) and not bs.empty else {}
                flags = bf["Flag"].tolist() if isinstance(bf, pd.DataFrame) and "Flag" in bf.columns else []
                payload["brsr"] = {
                    "risk_score": round(_safe_float(row.get("Overall_Risk_Score", 0)), 1),
                    "readiness_pct": round(_safe_float(row.get("Readiness_Score", 0)), 1),
                    "risk_overlay_bps": round(_safe_float(st.session_state.get("brsr_pd_adj", 0)) * 10000, 1),
                    "active_flags_top": flags[:6],
                    "water_cost_risk_5yr_cr": round(_safe_float(row.get("Water_Cost_Risk_Cr", 0)), 2),
                    "energy_transition_risk_cr": round(_safe_float(row.get("Energy_TR_Risk_Cr", 0)), 2),
                }

            if i_r:
                di = st.session_state.get("df_integrated_summary")
                if isinstance(di, pd.DataFrame) and not di.empty:
                    selected = []
                    for _, r in di.iterrows():
                        m = str(r.get("Metric", ""))
                        if any(k in m.lower() for k in ["integrated pd", "ecl", "dscr", "risk score", "capital"]):
                            selected.append({"metric": m, "value": str(r.get("Value", ""))})
                    payload["integrated"] = selected[:8]

            mc = st.session_state.get("mc_results")
            if mc:
                payload["monte_carlo"] = {
                    "mean_pd_pct": round(_safe_float(mc.get("Mean_PD", 0)) * 100, 2),
                    "pd_95_pct": round(_safe_float(mc.get("PD_95", 0)) * 100, 2),
                    "ecl_95_cr": round(_safe_float(mc.get("ECL_95", 0)), 2),
                }
            return payload

        def _advisor_system_prompt(name):
            return (
                "You are an expert climate-credit advisor for Indian companies. "
                "Use only the supplied numbers. Be specific, concise, and action-oriented. "
                "Do not reveal formulas, coefficients, hidden assumptions, or internal methodology. "
                "Limit output to practical business recommendations with numbers."
            )

        def _advisor_user_prompt(advisor_name, advisor_prefix, payload, extra='', focus=''):
            compact = _compact_json(payload, max_chars=3600)
            short_extra = _safe_text(extra, 700)
            short_focus = _safe_text(focus, 300)
            task_map = {
                "CFO": "Give 3 CFO actions with cost/benefit, timeline, and expected metric impact.",
                "Operations": "Give 3 operational risks and 3 mitigation actions.",
                "Strategy": "Give a 5-year roadmap with year-1, years-2-3, and years-4-5 priorities.",
                "Investor": "Give investor-readiness, market perception, and disclosure recommendations.",
                "Crisis": "Diagnose the biggest risks, trigger points, timing, and severity.",
            }
            low = advisor_name.lower()
            task = "Give concise risks, actions, owner, timeline, and expected impact."
            for key, val in task_map.items():
                if key.lower() in low:
                    task = val
                    break
            return _limit_prompt(f"""
Advisor: {advisor_name}
Task: {task}
Company: {company_name}; Sector: {sector}; Reporting year: {REPORTING_YEAR}
Focus: {short_focus or 'General decision support'}
Extra context: {short_extra or 'None'}
Data: {compact}

Output format:
1. Executive verdict: 2 sentences.
2. Key risks: max 3 bullets with numbers.
3. Recommended actions: max 3 bullets with owner, timeline, and expected impact.
4. Board message: 2 concise bullets.
Keep under 550 words.
""")

        # ── Advisor definitions ─────────────────────────────────────────
        ADVISORS = {
            "🔴 Crisis Diagnostician": {
                "icon": "🔴",
                "subtitle": "What is breaking, why, and how bad is it really",
                "color": C["coral"],
                "description": "Cuts through the numbers to tell you **exactly what is going wrong** in plain language — the real financial threats, the timeline, and what happens if nothing changes.",
                "system": """You are a blunt, experienced crisis diagnostic advisor combining deep expertise in corporate finance, credit risk, and climate economics. You have 25 years of experience restructuring distressed companies in carbon-intensive sectors across India and Southeast Asia.

YOUR JOB: Diagnose exactly what is going wrong with this company's finances under climate stress. Be specific, be direct, use real numbers from the data. No hedging. No regulatory jargon. Speak like a CFO who has seen companies fail.

STRUCTURE YOUR RESPONSE AS:

## 🔴 The Core Problem (2-3 sentences, brutally honest)
State the single biggest financial threat in plain language with exact numbers.

## 📉 The Cascade — What Breaks First, Then Next
Show the exact sequence: which metric hits the danger zone first, in which year, under which scenario. Use the year-by-year data. Be specific: "DSCR falls below covenant threshold of 1.2x in 2031 under Below 2°C, meaning the company cannot service ₹X Cr of debt without refinancing."

## 💣 The Three Landmines
The three specific things that could trigger a sudden credit event (not gradual deterioration). What triggers them. What year. What the impact would be on the bank's ECL.

## ⏰ How Much Time Is There
Give a realistic timeline: "Under current trajectory, the company has approximately X years before [specific crisis event] becomes likely."

## 🩺 Severity Verdict
Rate: [Manageable / Serious / Critical / Terminal] and explain exactly why in 3 sentences.

Use actual numbers from the data throughout. Do not use vague language like "may be impacted" — say "EBITDA will fall to ₹X Cr" or "PD increases from X% to Y%".""",
                "user_prefix": "Diagnose exactly what is going wrong with this company under climate stress. Use all numbers provided. Be direct and specific.",
            },

            "💰 CFO Action Advisor": {
                "icon": "💰",
                "subtitle": "The exact financial moves the CFO must make, with costs and timelines",
                "color": C["amber"],
                "description": "Gives the CFO a **prioritised action plan** with specific financial decisions, exact amounts, implementation timelines, and projected impact on PD, ECL, and DSCR.",
                "system": """You are a CFO advisor with 20 years of experience in capital allocation, debt restructuring, and climate transition finance at Indian conglomerates and PSU banks. You have overseen 15+ green bond issuances and climate CAPEX programmes.

YOUR JOB: Give the CFO a precise, implementable financial action plan based on the risk data. Every recommendation must include: what to do, how much it costs, how long it takes, and what it does to the key financial metrics.

STRUCTURE YOUR RESPONSE AS:

## 💰 CFO Priority #1: [Most Urgent Action — Name It Specifically]
**What:** Exactly what financial action to take (e.g., "Issue ₹X Cr green bonds to fund solar CAPEX")
**Cost/Benefit:** Exact cost, exact benefit to EBITDA/DSCR/PD
**Timeline:** Month-by-month implementation
**Risk if not done:** Specific consequence with numbers
**How to implement:** 3-4 concrete steps

## 💰 CFO Priority #2: [Second Most Urgent]
[Same structure]

## 💰 CFO Priority #3: [Third Priority]
[Same structure]

## 📊 Financial Impact Summary Table
Show before/after numbers for: DSCR, PD, ECL, Carbon Burden, CAPEX Gap if all three actions are implemented.

## ⚠️ What NOT To Do
2-3 common CFO mistakes in this situation that would make things worse. Be specific about why.

## 🏦 Financing Options Available
Specific instruments available for this sector in India: green bonds, REC monetisation, ESG-linked loans, SIDBI climate finance, IFC/ADB green facilities. Include approximate rates and ticket sizes.""",
                "user_prefix": "Give the CFO an exact, costed action plan to address these climate financial risks. Every action must have specific rupee amounts, timelines, and projected impact on metrics.",
            },

            "🏭 Operations & Engineering Advisor": {
                "icon": "🏭",
                "subtitle": "Exactly what to change in operations, with costs and carbon/cost impact",
                "color": C["mint"],
                "description": "Translates financial risk into **specific operational changes** — which equipment to replace, which processes to change, what the engineering costs are, and how much carbon and money each action saves.",
                "system": """You are an operations and engineering advisor specialising in decarbonisation of heavy industry in India. You have led 20+ energy transition projects at Steel, Power, Cement, and Oil & Gas plants. You know the specific Indian vendors, technology costs, and implementation timelines.

YOUR JOB: Turn the financial risk numbers into specific operational actions. The user needs to know exactly what to change on the ground — machinery, processes, energy sources — with real costs and real carbon/cost savings.

STRUCTURE YOUR RESPONSE AS:

## 🏭 Operational Root Cause
What is the specific operational driver of the company's climate financial risk? (e.g., "The core problem is reliance on coal-fired captive power at ₹X/kWh when grid + solar would cost ₹Y/kWh")

## ⚡ Quick Wins (0–12 months, under ₹50 Cr each)
List 3-5 specific operational changes:
- **Action:** [Specific change — e.g., "Install Variable Frequency Drives on X motors"]
- **Cost:** ₹X Cr
- **Annual saving:** ₹X Cr / X tCO₂e
- **Payback:** X months
- **Indian vendors:** [Specific companies]

## 🔧 Medium-Term Projects (1–3 years, ₹50–500 Cr)
List 3-4 specific capital projects with same detail as above.

## 🌱 Strategic Transformation (3–7 years, >₹500 Cr)
The 2-3 structural changes needed for long-term decarbonisation viability.

## 📐 CAPEX Prioritisation
Given the CAPEX gap from the model, which projects to do first to get the maximum reduction in PD and carbon burden per rupee spent.

## 🏗️ Implementation Sequencing
Month-by-month Gantt-style plan for Year 1. Be specific about what starts when.""",
                "user_prefix": "Translate these climate risk numbers into specific operational actions for this sector. Include exact costs, Indian vendors, and carbon/cost savings for each action.",
            },

            "📋 Regulatory Compliance Advisor": {
                "icon": "📋",
                "subtitle": "Exact compliance gaps, exact filing requirements, exact deadlines",
                "color": C["accent2"],
                "description": "Maps the company's current state against **every applicable Indian and global regulation** — what is non-compliant, what the penalties are, and the exact steps to achieve compliance.",
                "system": """You are a regulatory compliance specialist with deep expertise in Indian ESG and climate disclosure law: SEBI BRSR Core, RBI climate risk guidelines, Companies Act 2013 sustainability reporting, and ISSB IFRS S2. You have helped 40+ Indian listed companies achieve SEBI compliance and have appeared before RBI supervisory panels.

YOUR JOB: Tell the company exactly where it is non-compliant, what the specific regulatory consequences are, and give a precise compliance roadmap.

STRUCTURE YOUR RESPONSE AS:

## 🚨 Critical Non-Compliance Items (Act Now)
For each: The specific regulation, the specific gap, the specific penalty or consequence, and the deadline.

## ⚠️ Moderate Gaps (Address Within 6 Months)
Same format.

## ✅ Currently Compliant (Keep Doing)
Brief list of what is already in order.

## 📅 Regulatory Deadline Calendar
Month-by-month calendar of upcoming disclosure deadlines for this company:
- SEBI BRSR Core filing dates
- RBI climate stress test submission
- ICAAP climate overlay requirement
- ISSB S2 alignment dates

## 📝 Exact Disclosure Language
For each major gap, provide 2-3 sentences of **board-approved disclosure language** that accurately describes the risk without triggering regulatory concern. This is the exact text that can go into the annual report.

## 🏛️ Regulatory Engagement Strategy
Which regulators to proactively engage with, what to say, and how to get ahead of mandatory requirements rather than react to them.""",
                "user_prefix": "Identify every compliance gap against Indian and global climate regulations. Provide exact deadlines, penalties, and disclosure language that can be used directly in filings.",
            },

            "🏦 Banker & Credit Advisor": {
                "icon": "🏦",
                "subtitle": "How lenders see this company, what they will do, how to manage the relationship",
                "color": C["purple"],
                "description": "Tells you exactly **how your bank is thinking about this exposure** — whether they are likely to cut limits, raise spreads, or add covenants — and what to do to protect the credit relationship.",
                "system": """You are a senior credit risk advisor who has spent 18 years on both sides — as a corporate banker at SBI, HDFC Bank, and ICICI Bank, and as a credit risk consultant advising large Indian corporates on managing their bank relationships. You know exactly how Indian credit committees think about climate risk.

YOUR JOB: Tell the corporate exactly how their lending bank(s) are assessing this exposure right now, what credit actions they are likely to take, and how the corporate should manage the relationship.

STRUCTURE YOUR RESPONSE AS:

## 🏦 How Your Bank Is Reading These Numbers
Be specific: "A PD of X% puts this borrower in the [internal rating category] of most Indian banks. At this level, the relationship manager is likely to [specific action]."

## 📉 Credit Actions the Bank Is Likely to Take (Next 12–24 Months)
For each likely action: the trigger, the timeline, the financial impact on the company.
- Covenant review / tightening
- Margin overlay (specify basis points range)
- Limit reduction or non-renewal
- Enhanced monitoring / watch list
- Requirement for BRSR/climate disclosures

## 💬 What To Say To Your Banker (Right Now)
Exact talking points for the next banker meeting. What to proactively disclose, what narrative to lead with, what data to bring.

## 📄 Credit Documentation Strategy
Specific changes to make to the loan documentation at next renewal: covenants to negotiate, representations to include, green-linked pricing structures to propose.

## 🤝 Green Finance Opportunity
Based on these metrics, which green finance products is this company eligible for right now? Specify banks, schemes, approximate pricing vs conventional debt.

## ⚠️ What Would Trigger a Credit Crisis
The 3 specific scenarios (with numbers) that would cause the bank to move this to a stressed account. How far away is each trigger.""",
                "user_prefix": "Analyse how lending banks in India would assess this climate risk data. Tell the company exactly what credit actions to expect and how to manage the relationship.",
            },

            "📈 Investor & Equity Advisor": {
                "icon": "📈",
                "subtitle": "How markets will price this risk, what ESG investors want, how to improve valuation",
                "color": C["amber"],
                "description": "Analyses the **market and investor implications** — what ESG analysts will flag, how this affects equity valuation, what institutional investors require, and what the company must do to access green capital markets.",
                "system": """You are a capital markets and ESG investment advisor with 15 years experience covering Indian industrial stocks for FIIs, domestic mutual funds, and ESG-focused sovereign wealth funds. You have led 12 ESG-linked bond issuances and advised on 8 sustainability-linked equity stories.

YOUR JOB: Tell the company exactly how equity markets and institutional investors are assessing this climate risk, what it means for their valuation and cost of capital, and what they must do to access premium ESG capital.

STRUCTURE YOUR RESPONSE AS:

## 📊 Market Pricing of This Risk
How are equity investors likely to discount the valuation given these climate metrics? Translate PD and ECL numbers into EV/EBITDA multiple impact. Be specific: "A PD increase from X% to Y% under Below 2°C scenario typically leads institutional ESG analysts to apply a Z% discount to sector multiple."

## 🔍 What ESG Analysts Will Flag in Their Reports
The specific data points that trigger negative ESG analyst notes. For each: the metric, the threshold, the typical analyst language, the index exclusion risk.

## 💹 ESG Index Inclusion / Exclusion Risk
Specific indices (MSCI ESG, FTSE4Good, S&P BSE ESG, NIFTY100 ESG) — which ones this company risks exclusion from and why, based on the BRSR flags and climate risk scores.

## 🎯 What Institutional Investors Want (Prioritised List)
The 5 specific things the 10 largest institutional investors (FIIs + domestic MFs) in this sector are asking companies to do. For each: why they want it, what it signals, how to deliver it.

## 💰 Green Capital Raising Potential
Given these metrics: Can this company issue a green bond? At what premium/discount? What ESG improvements would unlock better pricing? Specific targets to hit.

## 📣 Investor Communication Strategy
What to say at the next earnings call, analyst day, and ESG roadshow. The specific metrics to highlight, the narrative arc, the forward guidance that will be credible.""",
                "user_prefix": "Analyse how equity markets and institutional investors will assess these climate risk metrics for this Indian company. Provide specific valuation impacts and an actionable investor strategy.",
            },

            "🗺️ Strategic Planning Advisor": {
                "icon": "🗺️",
                "subtitle": "5-year business transformation roadmap with milestones and investment decisions",
                "color": C["accent2"],
                "description": "Builds a **concrete 5-year strategic roadmap** that integrates climate risk mitigation with business growth — what to invest in, what to divest, how to restructure the business model to thrive rather than just survive.",
                "system": """You are a strategy advisor combining McKinsey-grade business strategy with deep climate transition expertise. You have led 5-year strategic transformations for 3 Indian steel companies, 2 power utilities, and 1 cement group navigating the energy transition. You understand exactly how to build a business case that wins board approval and delivers shareholder value while managing climate risk.

YOUR JOB: Build a practical 5-year strategic roadmap that turns this climate risk data into a business transformation plan. Not vague strategy — specific decisions, specific investments, specific milestones.

STRUCTURE YOUR RESPONSE AS:

## 🎯 Strategic Verdict
In 3 sentences: Is this company's core business model viable through 2035 under the most likely climate scenario? What is the fundamental strategic choice it faces?

## 🗺️ The 5-Year Transformation Roadmap

### Year 1 (Stabilise): Specific actions to stop the bleeding
- Financial: [Specific decisions with ₹ amounts]
- Operational: [Specific projects to start]
- Governance: [Specific structures to put in place]
- Success metrics: [What Year 1 must achieve — specific numbers]

### Years 2–3 (Reposition): Building the new business model
- Investment decisions: What to build, buy, or partner on
- Business lines to grow vs shrink
- Key hires and capabilities to build
- Milestone: [Specific financial/climate metrics by end of Year 3]

### Years 4–5 (Lead): Competitive differentiation through climate leadership
- Market positioning
- New revenue streams from decarbonisation
- Target: [Specific PD, carbon burden, and EBITDA targets by Year 5]

## ⚔️ Competitive Dynamics
How will climate regulation reshape competitive dynamics in this sector? Who wins, who loses, and what this company must do to be a winner.

## 💡 Hidden Opportunities
2-3 business opportunities created by climate transition that this company is positioned to capture, with revenue potential estimates.

## 🚫 Strategic Risks to Avoid
The 3 most dangerous strategic mistakes companies in this situation make. Be specific about why they are tempting and why they lead to failure.""",
                "user_prefix": "Build a 5-year strategic transformation roadmap for this company based on the climate risk data. Be specific about decisions, investments, and milestones.",
            },
        }


        # Override long internal advisor prompts with short public-safe prompts.
        # This materially reduces tokens and protects proprietary methodology.
        for _adv_name, _adv_cfg in ADVISORS.items():
            _adv_cfg["system"] = _advisor_system_prompt(_adv_name)
            _adv_cfg["user_prefix"] = _adv_cfg.get("user_prefix", "Provide concise decision support.")[:220]

        # ── UI Layout ───────────────────────────────────────────────────
        st.markdown(f"""
        <div style="background:{C['card']};border:1px solid {C['bg_ocean']};border-radius:10px;
                    padding:14px 18px;margin-bottom:16px;">
            <div style="font-size:13px;color:{C['accent2']};font-weight:600;margin-bottom:6px;">
                How to use these advisors
            </div>
            <div style="font-size:12px;color:{C['slate']};line-height:1.7;">
                Each advisor looks at the <strong style="color:{C['white']};">same model results</strong>
                through a completely different lens.
                Run the one most relevant to your immediate need, or run all seven and compare.
                Every response uses your <strong style="color:{C['white']};">actual computed numbers</strong>
                — not generic advice.
            </div>
        </div>""", unsafe_allow_html=True)

        # Advisor selector
        advisor_names = list(ADVISORS.keys())
        selected_advisor = st.radio(
            "Choose your advisor",
            advisor_names,
            horizontal=False,
            label_visibility="collapsed",
        )

        adv = ADVISORS[selected_advisor]

        # Show advisor card
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{C['bg_mid']},{C['card']});
                    border:1px solid {adv['color']};border-left:5px solid {adv['color']};
                    border-radius:10px;padding:16px 20px;margin:12px 0;">
            <div style="font-size:18px;font-weight:700;color:{adv['color']};margin-bottom:4px;">
                {selected_advisor}
            </div>
            <div style="font-size:12px;color:{C['accent2']};font-weight:600;margin-bottom:8px;">
                {adv['subtitle']}
            </div>
            <div style="font-size:12px;color:{C['slate']};line-height:1.6;">
                {adv['description']}
            </div>
        </div>""", unsafe_allow_html=True)

        # Optional context box
        with st.expander("➕ Add context to improve the response (optional)", expanded=False):
            extra_ctx = st.text_area(
                "Anything the advisor should know that's not in the model data",
                placeholder=(
                    "E.g.: We have a board-approved ₹800 Cr green CAPEX plan starting Q3 2026. "
                    "Our main lender is SBI with a loan renewal due in March 2027. "
                    "We are targeting MSCI ESG inclusion by end 2025. "
                    "Our main competitor JSW Steel has already committed to net zero by 2050."
                ),
                height=100,
                key="ai_extra_ctx",
            )
            focus_area = st.text_input(
                "Specific question or focus area (optional)",
                placeholder="E.g.: Focus specifically on what to do about the CAPEX gap and how to finance it.",
                key="ai_focus",
            )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            run_btn = st.button(
                f"🧠 Ask the {adv['icon']} {selected_advisor.split(' ', 1)[1]}",
                type="primary",
                width="stretch",
            )
        with col_btn2:
            clear_btn = st.button("🗑️ Clear Output", width="stretch")

        if clear_btn:
            if "ai_outputs" in st.session_state:
                st.session_state["ai_outputs"] = {}
            st.rerun()

        # ── Run the selected advisor ────────────────────────────────────
        if run_btn:
            payload = _build_compact_payload()
            extra = st.session_state.get("ai_extra_ctx", "").strip()
            focus = st.session_state.get("ai_focus", "").strip()
            user_msg = _advisor_user_prompt(selected_advisor, adv.get("user_prefix", ""), payload, extra, focus)

            st.caption(f"AI request size: {len(user_msg):,} characters · compact mode enabled")
            output_placeholder = st.empty()
            output = ""

            with st.spinner(f"Your {selected_advisor} is analysing the compact risk summary..."):
                try:
                    client = _get_groq_client()
                    stream = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": adv["system"]},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.25,
                        max_tokens=AI_MAX_OUTPUT_TOKENS,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = getattr(chunk.choices[0].delta, "content", None)
                        if delta:
                            output += delta
                            output_placeholder.markdown(output + "▌")
                    output_placeholder.markdown(output)
                    if "ai_outputs" not in st.session_state:
                        st.session_state["ai_outputs"] = {}
                    st.session_state["ai_outputs"][selected_advisor] = output
                    st.session_state["ai_outputs"]["_last_advisor"] = selected_advisor
                except Exception as e:
                    err = str(e)
                    if "413" in err or "Request too large" in err or "rate_limit" in err or "TPM" in err:
                        st.warning("AI request was still too large for the current Groq tier. Showing compact fallback insight instead.")
                        fallback = (
                            f"### {selected_advisor} — Compact fallback\n\n"
                            f"Groq rejected the request due to the current token-per-minute limit, even after compression. "
                            f"Use the numeric dashboards for decisioning, or run one advisor at a time after a short pause.\n\n"
                            f"**Compact data snapshot:** `{_compact_json(payload, max_chars=1200)}`"
                        )
                        st.session_state.setdefault("ai_outputs", {})[selected_advisor] = fallback
                        st.session_state["ai_outputs"]["_last_advisor"] = selected_advisor
                        output_placeholder.markdown(fallback)
                    else:
                        st.error(f"AI error: {e}. Check that GROQ_API_KEY is set in Streamlit Secrets or your environment.")

        # ── Display output ──────────────────────────────────────────────
        ai_outs = st.session_state.get("ai_outputs", {})
        last_adv = ai_outs.get("_last_advisor", selected_advisor)
        output_text = ai_outs.get(selected_advisor, "")

        if output_text:
            # Output card
            st.markdown(f"""
            <div style="background:{C['bg_dark']};border:1px solid {ADVISORS[selected_advisor]['color']};
                        border-radius:10px;padding:4px 4px 4px 4px;margin-top:8px;">
                <div style="background:{C['card']};border-radius:8px;padding:20px 24px;">
            """, unsafe_allow_html=True)
            st.markdown(output_text)
            st.markdown("</div></div>", unsafe_allow_html=True)

            # Copy / download button
            st.download_button(
                label="⬇️ Download this analysis (.txt)",
                data=output_text,
                file_name=f"ICCRE_{company_name.replace(' ','_')}_{selected_advisor.split(' ',1)[1].replace(' ','_')[:30]}_{REPORTING_YEAR}.txt",
                mime="text/plain",
                width="content",
            )

        elif not run_btn:
            st.markdown(f"""
            <div style="background:{C['card']};border:1px dashed {C['bg_ocean']};border-radius:10px;
                        padding:32px;text-align:center;margin-top:8px;">
                <div style="font-size:36px;margin-bottom:8px;">{adv['icon']}</div>
                <div style="font-size:14px;color:{C['slate']};">
                    Click the button above to get your analysis from the {selected_advisor}.
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Previous outputs from other advisors ────────────────────────
        other_outputs = {k: v for k, v in ai_outs.items()
                         if k != selected_advisor and k != "_last_advisor" and v}
        if other_outputs:
            st.divider()
            st.markdown(f"<div style='font-size:12px;color:{C['slate']};font-weight:600;"
                        f"text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>"
                        f"Previously generated analyses</div>", unsafe_allow_html=True)
            for prev_name, prev_text in other_outputs.items():
                prev_adv = ADVISORS.get(prev_name, {})
                with st.expander(f"{prev_name} — click to expand"):
                    st.markdown(prev_text)
                    st.download_button(
                        label=f"⬇️ Download {prev_name.split(' ',1)[1][:25]}",
                        data=prev_text,
                        file_name=f"ICCRE_{company_name.replace(' ','_')}_{prev_name.split(' ',1)[1].replace(' ','_')[:30]}_{REPORTING_YEAR}.txt",
                        mime="text/plain",
                        key=f"dl_{prev_name[:20]}",
                    )

    # ============================================================
    # TAB 8 — METHODOLOGY
    # ============================================================
    # ============================================================
    # TAB 8 — METHODOLOGY  (auditor-appropriate, IP-protected)
    # ============================================================
with methodology_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🛡️ Model Governance</h2>", unsafe_allow_html=True)
    st.caption("Public demo governance view · IP-protected · fictional company only")
    st.info(PUBLIC_DEMO_NOTE)

    scope_badge("single", f"Reporting-year snapshot results use the locked Bharat Steel Industries Ltd inputs for {REPORTING_YEAR}.")
    scope_badge("multi", "Scenario charts are forward-looking NGFS pathway projections across selected years.")

    st.markdown("""
    **What this demo shows**
    - How climate transition pathways can be translated into financial stress indicators.
    - How BRSR-style operational signals can support a governance/readiness risk view.
    - How physical exposure screening can be linked to asset-level business interruption.
    - How board-level risk summaries can be produced for decision support.

    **What this public demo does not expose**
    - Proprietary equations and coefficient values.
    - Full calibration workflow and sector parameter registry.
    - Full Excel model export and raw intermediate tables.
    - Client-specific validation and governance pack.
    """)

    st.warning(
        "This is a decision-support and scenario-analysis demo using fictional data. It is not financial, credit, investment, legal, or regulatory advice. "
        "Production use requires independent review, calibration, and validation."
    )

with validation_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🧪 Pilot Readiness</h2>", unsafe_allow_html=True)
    st.caption("What serious users can evaluate in a private pilot")

    st.markdown("""
    **Recommended pilot validation checklist**

    | Area | What we validate in pilot access |
    |---|---|
    | Data fit | Company financials, emissions, BRSR inputs, asset register |
    | Scenario relevance | NGFS scenario selection and reporting-year assumptions |
    | Credit-risk reasonableness | PD, ECL, DSCR and capital-signal interpretation |
    | Physical-risk screening | Asset exposure, hazard layers, revenue-loss reasonableness |
    | Governance | Model-use limitations, audit trail, and board-report wording |

    Public demo validation is intentionally limited because the demo uses fictional locked data.
    """)

    st.success("For banks, NBFCs, consultants and investors: request a pilot to test your own company or portfolio data.")

with calibration_tab:
    st.markdown(f"<h2 style='color:{C['white']}'>🔒 Enterprise Preview</h2>", unsafe_allow_html=True)
    st.caption("Advanced features available only in private pilot / enterprise access")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **Calibration Pack**
        - Sector parameter tuning
        - Borrower-specific assumptions
        - Historical comparison
        - Challenger-model review
        """)
    with c2:
        st.markdown("""
        **Portfolio Pack**
        - Multi-company upload
        - Sector concentration
        - Climate VaR
        - Board dashboard
        """)
    with c3:
        st.markdown("""
        **Governance Pack**
        - Model documentation
        - Audit trail
        - Scenario memo
        - Exportable board report
        """)

    st.markdown(f"""
    <div style="background:{C['card']};border:1px solid {C['accent2']};border-radius:10px;padding:16px;margin-top:12px;">
      <div style="font-size:16px;font-weight:700;color:{C['accent2']};">Request private pilot access</div>
      <div style="font-size:12px;color:{C['slate']};margin-top:6px;">Email {CONTACT_EMAIL} with your organisation, use case and preferred pilot scope.</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# GET ACCESS TAB — public demo CTA
# ============================================================
with access_tab:
    import urllib.parse

    st.markdown(f"<h2 style='color:{C['white']};margin-bottom:4px;'>🚀 Request Pilot Access</h2>", unsafe_allow_html=True)
    st.caption("Move from fictional demo to your own company / portfolio analysis")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['bg_mid']},{C['card']});border:1px solid {C['accent']};border-radius:12px;padding:24px 28px;margin-bottom:20px;">
      <div style="font-size:18px;font-weight:700;color:{C['white']};margin-bottom:6px;">What unlocks in a private pilot</div>
      <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:14px;">
        <div style="flex:1;min-width:210px;background:{C['bg_dark']};border:1px solid {C['bg_ocean']};border-radius:10px;padding:14px;">
          <div style="font-size:14px;font-weight:700;color:{C['accent2']};">Custom company analysis</div>
          <div style="font-size:11px;color:{C['slate']};line-height:1.7;margin-top:6px;">Use your own financials, emissions, asset register and BRSR inputs.</div>
        </div>
        <div style="flex:1;min-width:210px;background:{C['bg_dark']};border:1px solid {C['bg_ocean']};border-radius:10px;padding:14px;">
          <div style="font-size:14px;font-weight:700;color:{C['mint']};">Portfolio view</div>
          <div style="font-size:11px;color:{C['slate']};line-height:1.7;margin-top:6px;">Assess multiple borrowers, sectors and exposure concentrations.</div>
        </div>
        <div style="flex:1;min-width:210px;background:{C['bg_dark']};border:1px solid {C['bg_ocean']};border-radius:10px;padding:14px;">
          <div style="font-size:14px;font-weight:700;color:{C['amber']};">Governance pack</div>
          <div style="font-size:11px;color:{C['slate']};line-height:1.7;margin-top:6px;">Get full report exports, documentation, assumptions review and board-ready outputs.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Best-fit pilot users**
    - Banks / NBFCs evaluating climate-adjusted credit risk
    - ESG and climate-risk consultants building client reports
    - CFO / treasury teams assessing transition and physical exposure
    - Investors and accelerators evaluating climate-finance infrastructure
    """)

    with st.form("pilot_request_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        req_name = c1.text_input("Name", value=st.session_state.get("demo_lead", {}).get("name", ""))
        req_email = c2.text_input("Work Email", value=st.session_state.get("demo_lead", {}).get("email", ""))
        req_org = c1.text_input("Organisation", value=st.session_state.get("demo_lead", {}).get("organisation", ""))
        req_use = c2.selectbox("Pilot use case", ["Bank / NBFC portfolio", "Single company analysis", "ESG consulting", "Investor diligence", "Partnership", "Other"])
        req_msg = st.text_area("What would you like to evaluate in a pilot?", height=100)
        submit_req = st.form_submit_button("Send Pilot Request", type="primary", width="stretch")

    if submit_req:
        lead = {
            "name": req_name.strip(),
            "email": req_email.strip(),
            "organisation": req_org.strip(),
            "role": st.session_state.get("demo_lead", {}).get("role", ""),
            "purpose": f"Pilot request: {req_use}",
            "notes": req_msg.strip(),
            "demo_company": DEMO_DATASET["company_name"],
            "model_version": MODEL_VERSION,
        }
        if not _valid_email(req_email):
            st.error("Please enter a valid work email.")
        elif not req_org.strip():
            st.error("Please enter your organisation.")
        else:
            _save_demo_lead(lead)
            st.success("Pilot request recorded. We will use the submitted contact details for follow-up.")
            mailto = f"mailto:{CONTACT_EMAIL}?subject=ICCRE Pilot Request&body=" + urllib.parse.quote(
                f"Name: {req_name}\nEmail: {req_email}\nOrganisation: {req_org}\nUse case: {req_use}\n\n{req_msg}"
            )
            st.markdown(f"[Open email draft to ICCRE]({mailto})")

    st.warning("Public demo uses fictional data and is not financial, credit, investment, legal or regulatory advice.")
