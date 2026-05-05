import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import urllib.request
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go

# ============================================================
# ICCRE LINKEDIN DEMO APP
# Purpose: Public demo with lead capture, limited features, locked fictional data
# ============================================================
MODEL_VERSION = "LinkedIn_Demo_v1.0"
PRODUCT_TAGLINE = "India-focused climate-to-credit risk demo"
CONTACT_EMAIL = "hello@iccre.in"
PRODUCT_URL = "https://iccre.in"

st.set_page_config(
    page_title="ICCRE Demo — Climate-to-Credit Risk Engine",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DEMO CONFIG
# ============================================================
DEMO = {
    "company_name": "Bharat Steel Industries Ltd",
    "sector": "Steel",
    "reporting_year": 2035,
    "revenue_0": 18000.0,          # ₹ Cr
    "ebitda_margin_0": 0.22,
    "interest_payment": 1100.0,    # ₹ Cr
    "ead": 12000.0,                # ₹ Cr
    "scope1": 4200000.0,
    "scope2": 1800000.0,
    "scope3": 2000000.0,
    "high_carbon_assets": 8000.0,  # ₹ Cr
    "base_pd": 0.015,
    "lgd_0": 0.45,
    "carbon_pass_through": 0.35,
    "demand_elasticity": -0.40,
    "planned_capex": 1800.0,
    "abatement_cost": 4500.0,
    "abatement_potential": 0.30,
}

YEARS = [2025, 2030, 2035, 2040]
SCENARIO_WEIGHTS = {
    "Current Policies": 0.50,
    "Nationally Determined Contributions (NDCs)": 0.30,
    "Net Zero 2050": 0.20,
}
SCENARIO_COLORS = {
    "Current Policies": "#FF6B6B",
    "Nationally Determined Contributions (NDCs)": "#FFD166",
    "Net Zero 2050": "#22D3EE",
}

# Hidden demo assumptions. Do not show these in public UI.
USD_INR = 83.0
CARBON_PRICE_INFLATION_FACTOR = 1.38
PD_FLOOR = DEMO["base_pd"]
PD_CAP = 0.35
BASE_CREDIT_SPREAD = 0.02
SECTOR_PARAMS = {
    "gdp_sensitivity": 1.20,
    "alpha_dscr": 1.10,
    "beta_carbon_credit": 1.40,
    "spread_sensitivity": 1.40,
}
BRSR_PD_SPREAD = {
    "Low renewable energy": 0.0018,
    "No verified emissions data": 0.0025,
    "Low target coverage": 0.0030,
}
BRSR_MAX_MULTIPLIER = 1.20

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; }
.stApp { background: linear-gradient(145deg, #062F2E 0%, #0B4D4B 55%, #115E6D 100%); background-attachment: fixed; }
h1,h2,h3,h4,h5,h6,p,span,div,label { color: #E2E8F0 !important; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.demo-hero { background: linear-gradient(135deg, rgba(13,59,74,.96), rgba(11,77,75,.96)); border: 1px solid #115E6D; border-radius: 18px; padding: 28px 30px; margin-bottom: 16px; box-shadow: 0 10px 30px rgba(0,0,0,.20); }
.demo-eyebrow { color: #22D3EE !important; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: .10em; }
.demo-title { color: #FFFFFF !important; font-size: 32px; line-height: 1.1; font-weight: 800; letter-spacing: -0.04em; margin-top: 8px; }
.demo-subtitle { color: #94A3B8 !important; font-size: 14px; line-height: 1.6; margin-top: 10px; max-width: 900px; }
.demo-card { background: linear-gradient(135deg, #0D3B4A 0%, #0B4D4B 100%); border: 1px solid #115E6D; border-radius: 14px; padding: 17px 15px; min-height: 125px; margin-bottom: 12px; display: flex; flex-direction: column; justify-content: center; gap: 6px; }
.card-label { color: #94A3B8 !important; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; font-weight: 800; line-height: 1.25; }
.card-value { color: #22D3EE !important; font-family: 'IBM Plex Mono', monospace !important; font-size: clamp(22px, 2.5vw, 32px); font-weight: 800; line-height: 1.05; overflow-wrap: anywhere; }
.card-note { color: #94A3B8 !important; font-size: 11px; line-height: 1.4; }
.locked-card { background: rgba(2, 6, 23, .35); border: 1px dashed #22D3EE; border-radius: 14px; padding: 18px; min-height: 116px; display:flex; flex-direction:column; justify-content:center; }
.locked-title { color: #E6F1F5 !important; font-size: 13px; font-weight: 800; }
.locked-text { color: #94A3B8 !important; font-size: 12px; margin-top: 6px; }
.notice { background: rgba(255,209,102,.10); border-left: 4px solid #FFD166; border-radius: 10px; padding: 12px 14px; color: #E6F1F5 !important; font-size: 13px; line-height: 1.5; margin: 10px 0 15px; }
.success-note { background: rgba(128,237,153,.10); border-left: 4px solid #80ED99; border-radius: 10px; padding: 12px 14px; color: #E6F1F5 !important; font-size: 13px; line-height: 1.5; }
.stButton > button { background: linear-gradient(135deg, #06B6D4, #22D3EE) !important; color: #062F2E !important; border: none !important; border-radius: 10px !important; font-weight: 800 !important; }
.stDownloadButton > button { background: #0D3B4A !important; color: #22D3EE !important; border: 1px solid #115E6D !important; border-radius: 10px !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(6,47,46,.60) !important; border-bottom: 1px solid #115E6D; }
.stTabs [data-baseweb="tab"] { color: #94A3B8 !important; font-size: 13px !important; }
.stTabs [aria-selected="true"] { color: #22D3EE !important; background: #0D3B4A !important; border-radius: 9px 9px 0 0; }
input, textarea, .stSelectbox [data-baseweb="select"] > div { background-color: #0D3B4A !important; color: #E2E8F0 !important; border: 1px solid #115E6D !important; border-radius: 9px !important; }
hr { border-color: #115E6D !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================
def fmt_money(v, decimals=1):
    try:
        return f"₹{float(v):,.{decimals}f} Cr"
    except Exception:
        return "—"


def fmt_pct(v, decimals=2):
    try:
        return f"{float(v):.{decimals}%}"
    except Exception:
        return "—"


def fmt_num(v, suffix="", decimals=2):
    try:
        return f"{float(v):,.{decimals}f}{suffix}"
    except Exception:
        return "—"


def logit(p):
    p = np.clip(float(p), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def sigmoid(x):
    x = np.clip(float(x), -50, 50)
    return 1 / (1 + np.exp(-x))


def ecl_cr(pd, lgd, ead_cr):
    return float(np.clip(pd, 0, 1) * np.clip(lgd, 0, 1) * max(float(ead_cr), 0))


def brsr_multiplier(brsr_signal):
    if brsr_signal <= 0:
        return 1.0
    return float(np.clip(1.0 + (brsr_signal / 0.015) * (BRSR_MAX_MULTIPLIER - 1.0), 1.0, BRSR_MAX_MULTIPLIER))


def metric_card(label, value, note="", color="#22D3EE"):
    st.markdown(f"""
    <div class="demo-card">
      <div class="card-label">{label}</div>
      <div class="card-value" style="color:{color} !important;">{value}</div>
      <div class="card-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def locked_card(title, text="Available in full pilot access"):
    st.markdown(f"""
    <div class="locked-card">
      <div class="locked-title">🔒 {title}</div>
      <div class="locked-text">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def save_lead(payload):
    Path("Data").mkdir(exist_ok=True)
    file_path = Path("Data/demo_leads.csv")
    row = pd.DataFrame([{**payload, "timestamp_utc": datetime.utcnow().isoformat()}])
    row.to_csv(file_path, mode="a", header=not file_path.exists(), index=False)

    # Optional: set LEADS_WEBHOOK_URL in Streamlit Secrets to post lead data to Google Apps Script / Make / Zapier.
    try:
        webhook = st.secrets.get("LEADS_WEBHOOK_URL")
    except Exception:
        webhook = os.getenv("LEADS_WEBHOOK_URL")
    if webhook:
        try:
            req = urllib.request.Request(
                webhook,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass


@st.cache_data
def load_ngfs_public():
    candidates = [Path("Data/ngfs_scenarios.csv"), Path("data/ngfs_scenarios.csv"), Path("ngfs_scenarios.csv")]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)
            for c in df.select_dtypes("object"):
                df[c] = df[c].astype(str).str.strip()
            year_cols = [c for c in df.columns if str(c).isdigit() and int(c) in YEARS]
            keep = df["Variable"].astype(str).str.contains("Carbon|GDP|Surface Temperature|Temperature", case=False, na=False)
            df = df.loc[keep]
            if not year_cols or df.empty:
                break
            return df.melt(
                id_vars=[c for c in df.columns if c not in year_cols],
                value_vars=year_cols,
                var_name="Year",
                value_name="Value",
            ).assign(Year=lambda x: x["Year"].astype(int))

    # Fallback synthetic public-demo scenario data so the demo does not fail if NGFS file is missing.
    records = []
    carbon = {
        "Current Policies": [15, 18, 21, 24],
        "Nationally Determined Contributions (NDCs)": [20, 55, 90, 125],
        "Net Zero 2050": [30, 125, 260, 390],
    }
    gdp = {
        "Current Policies": [100, 112, 126, 141],
        "Nationally Determined Contributions (NDCs)": [100, 110, 121, 133],
        "Net Zero 2050": [100, 106, 111, 118],
    }
    temp = {
        "Current Policies": [1.35, 1.50, 1.68, 1.88],
        "Nationally Determined Contributions (NDCs)": [1.35, 1.47, 1.60, 1.70],
        "Net Zero 2050": [1.35, 1.42, 1.48, 1.50],
    }
    for scen in carbon:
        for year, cp, gd, tm in zip(YEARS, carbon[scen], gdp[scen], temp[scen]):
            records += [
                {"Scenario": scen, "Variable": "Price|Carbon", "Year": year, "Value": cp},
                {"Scenario": scen, "Variable": "GDP|MER", "Year": year, "Value": gd},
                {"Scenario": scen, "Variable": "Surface Temperature", "Year": year, "Value": tm},
            ]
    return pd.DataFrame(records)


def get_value(df_long, scenario, year, keyword, default=0.0):
    rows = df_long[(df_long["Scenario"] == scenario) & (df_long["Year"] == year) & df_long["Variable"].astype(str).str.contains(keyword, case=False, na=False)]
    if rows.empty:
        return default
    return float(rows["Value"].iloc[0])


def run_transition_demo(selected_scenarios):
    df_long = load_ngfs_public()
    total_emissions = DEMO["scope1"] + DEMO["scope2"] + DEMO["scope3"]
    results = []
    for scen in selected_scenarios:
        for y in YEARS:
            cp = get_value(df_long, scen, y, "Carbon", 20.0)
            temp = get_value(df_long, scen, y, "Temperature", 1.5)
            gdp = get_value(df_long, scen, y, "GDP", 100.0)
            gdp_base = get_value(df_long, "Current Policies", y, "GDP", gdp)
            gdp_shock = (gdp - gdp_base) / max(gdp_base, 1e-6)

            carbon_price_adj = cp * CARBON_PRICE_INFLATION_FACTOR
            gross_carbon_cost = total_emissions * carbon_price_adj * USD_INR / 1e7
            net_carbon_cost = gross_carbon_cost * (1 - DEMO["carbon_pass_through"])

            revenue = DEMO["revenue_0"] * (1 + SECTOR_PARAMS["gdp_sensitivity"] * gdp_shock)
            revenue *= (1 + DEMO["demand_elasticity"] * (net_carbon_cost / max(DEMO["revenue_0"], 1)))
            physical_loss = 0.02 * max(0, temp - 1.5)
            revenue *= (1 - physical_loss)
            revenue_floor = DEMO["revenue_0"] * 0.05
            non_viable = revenue < revenue_floor
            revenue = max(revenue, revenue_floor)

            years_from_start = max(0, y - 2025)
            margin_t = max(0.10, DEMO["ebitda_margin_0"] * (1 - 0.005 * years_from_start))
            ebitda = revenue * margin_t - net_carbon_cost
            carbon_burden = net_carbon_cost / max(revenue, 1)
            credit_spread = BASE_CREDIT_SPREAD * (1 + SECTOR_PARAMS["spread_sensitivity"] * carbon_burden)
            dscr = ebitda / max(DEMO["interest_payment"] * (1 + credit_spread), 1e-6)
            dscr_gap = np.clip(1.5 - dscr, -4.0, 6.0)

            stressed_pd = sigmoid(logit(DEMO["base_pd"]) + SECTOR_PARAMS["alpha_dscr"] * dscr_gap + SECTOR_PARAMS["beta_carbon_credit"] * carbon_burden)
            pd_t = np.clip(max(DEMO["base_pd"], stressed_pd), DEMO["base_pd"], PD_CAP)
            if non_viable:
                pd_t = PD_CAP
            lgd_t = np.clip(DEMO["lgd_0"] * (1 + 0.20 * carbon_burden + 0.30 * physical_loss), 0, 1)
            ecl = ecl_cr(pd_t, lgd_t, DEMO["ead"])

            stranded_ratio = 0 if carbon_price_adj < 50 else min((carbon_price_adj - 50) / 200, 1.0)
            stranded_assets = DEMO["high_carbon_assets"] * min(stranded_ratio, 0.80)

            results.append({
                "Scenario": scen,
                "Year": y,
                "Revenue": revenue,
                "EBITDA": ebitda,
                "DSCR": dscr,
                "Carbon_Burden": carbon_burden,
                "PD": pd_t,
                "LGD": lgd_t,
                "ECL": ecl,
                "Stranded_Assets": stranded_assets,
                "Business_NonViable": bool(non_viable),
            })
    return pd.DataFrame(results)


def run_brsr_demo(renewable_share, verified_data, target_coverage):
    flags = []
    if renewable_share < 20:
        flags.append("Low renewable energy")
    if not verified_data:
        flags.append("No verified emissions data")
    if target_coverage < 50:
        flags.append("Low target coverage")
    signal = float(np.clip(sum(BRSR_PD_SPREAD.get(f, 0) for f in flags), 0, 0.015))
    readiness_items = [renewable_share >= 20, verified_data, target_coverage >= 50, True, False]
    readiness = sum(1 for x in readiness_items if x) / len(readiness_items) * 100
    score = 0
    if renewable_share < 20:
        score += 12
    if not verified_data:
        score += 12
    if target_coverage < 50:
        score += 10
    score += 8  # no board oversight exposed in public demo
    return {
        "flags": flags,
        "signal": signal,
        "readiness": readiness,
        "score": min(100, score),
    }


def build_integrated(df_t, brsr_signal):
    report_year = DEMO["reporting_year"]
    df_report = df_t[df_t["Year"] == report_year]
    worst_report = df_report.loc[df_report["PD"].idxmax()]
    peak = df_t.loc[df_t["PD"].idxmax()]

    brsr_mult = brsr_multiplier(brsr_signal)
    integrated_pd = float(np.clip(worst_report["PD"] * brsr_mult, DEMO["base_pd"], PD_CAP))
    integrated_lgd = float(max(worst_report["LGD"], DEMO["lgd_0"]))
    integrated_ecl = ecl_cr(integrated_pd, integrated_lgd, DEMO["ead"])
    ecl_ead = integrated_ecl / DEMO["ead"]
    cap_signal = "Severe" if ecl_ead >= 0.05 else "Elevated" if ecl_ead >= 0.03 else "Moderate" if ecl_ead >= 0.015 else "Limited"
    risk_score = min(100, round(0.45 * min(100, integrated_pd / 0.20 * 100) + 0.45 * min(100, ecl_ead / 0.05 * 100) + 10, 1))
    first_non_viable = df_t[df_t["Business_NonViable"]]["Year"].min() if df_t["Business_NonViable"].any() else None
    return {
        "report_worst": worst_report,
        "peak": peak,
        "integrated_pd": integrated_pd,
        "integrated_ecl": integrated_ecl,
        "capital_signal": cap_signal,
        "risk_score": risk_score,
        "first_non_viable": first_non_viable,
        "brsr_mult": brsr_mult,
    }


def plot_line(df, y, title, tickformat=None, money=False):
    fig = go.Figure()
    for scen in df["Scenario"].unique():
        d = df[df["Scenario"] == scen]
        fig.add_trace(go.Scatter(
            x=d["Year"], y=d[y], mode="lines+markers", name=scen,
            line=dict(color=SCENARIO_COLORS.get(scen, "#80ED99"), width=2.6),
            marker=dict(size=7),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#E6F1F5", size=15)),
        height=320,
        plot_bgcolor="#062F2E",
        paper_bgcolor="#062F2E",
        font=dict(color="#E6F1F5", size=11),
        margin=dict(l=45, r=20, t=55, b=45),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#0B4D4B", zeroline=False)
    fig.update_yaxes(gridcolor="#0B4D4B", zeroline=False)
    if tickformat:
        fig.update_yaxes(tickformat=tickformat)
    if money:
        fig.update_yaxes(title="₹ Cr")
    return fig

# ============================================================
# LEAD CAPTURE GATE
# ============================================================
if "lead_submitted" not in st.session_state:
    st.session_state.lead_submitted = False

if not st.session_state.lead_submitted:
    st.markdown(f"""
    <div class="demo-hero">
      <div class="demo-eyebrow">ICCRE Public Demo</div>
      <div class="demo-title">Climate-to-credit risk engine for Indian financial decision makers</div>
      <div class="demo-subtitle">
        This limited demo uses fictional data to show how climate transition scenarios and BRSR signals can be translated into PD, ECL, DSCR, capital stress, and board-style insights.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="notice">Please share a few details to access the limited demo. This keeps the demo controlled and helps us prioritise serious pilot requests.</div>', unsafe_allow_html=True)

    with st.form("lead_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            email = st.text_input("Work Email ID *", placeholder="name@organisation.com")
            org = st.text_input("Organisation *", placeholder="Bank / NBFC / Corporate / Consulting firm")
        with c2:
            purpose = st.selectbox("Purpose *", ["Evaluate for bank / NBFC use", "Corporate climate-risk assessment", "ESG / BRSR advisory", "Academic / research", "Partnership discussion", "Other"])
            role = st.text_input("Role / Designation", placeholder="Risk, ESG, Finance, Product, Founder...")
        consent = st.checkbox("I understand this demo uses fictional data and is not financial, credit, investment, or regulatory advice.")
        submitted = st.form_submit_button("Access limited demo", type="primary")

    if submitted:
        if not email or "@" not in email or not org or not purpose or not consent:
            st.error("Please enter a valid email, organisation, purpose, and accept the demo note.")
        else:
            lead = {"email": email.strip(), "organisation": org.strip(), "purpose": purpose, "role": role.strip()}
            st.session_state.lead_submitted = True
            st.session_state.lead = lead
            save_lead(lead)
            st.rerun()
    st.stop()

# ============================================================
# HEADER AFTER ACCESS
# ============================================================
st.markdown(f"""
<div class="demo-hero">
  <div style="display:flex;justify-content:space-between;gap:16px;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <div class="demo-eyebrow">ICCRE Limited Demo · {MODEL_VERSION}</div>
      <div class="demo-title">Climate-to-credit risk engine</div>
      <div class="demo-subtitle">
        Fictional demo company: <b>{DEMO['company_name']}</b> · Sector: <b>{DEMO['sector']}</b> · Reporting year: <b>{DEMO['reporting_year']}</b>.
        The public demo shows only selected outputs. Advanced calibration, physical GIS, portfolio analytics, detailed methodology, and full exports are available in pilot access.
      </div>
    </div>
    <div style="min-width:180px;text-align:right;">
      <div style="color:#94A3B8;font-size:11px;text-transform:uppercase;font-weight:800;">Logged access</div>
      <div style="color:#22D3EE;font-size:13px;font-weight:700;">{st.session_state.lead.get('organisation','')}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="notice"><b>Demo limitation:</b> Fictional data only. This is a scenario-analysis product demo, not credit advice or a regulatory model. Full methodology, calibration, exports, and custom inputs are intentionally hidden in this public version.</div>', unsafe_allow_html=True)

# Public-safe controls
with st.container():
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        selected_scenarios = st.multiselect(
            "Scenario view",
            list(SCENARIO_WEIGHTS.keys()),
            default=list(SCENARIO_WEIGHTS.keys()),
            help="Public demo scenario list is limited. Full pilot supports custom configuration.",
        )
    with c2:
        renewable_share = st.slider("Renewable energy", 0, 60, 15, help="Limited BRSR demo input")
    with c3:
        verified_data = st.toggle("Verified data", value=False, help="Limited BRSR demo input")

if not selected_scenarios:
    st.warning("Select at least one scenario to run the demo.")
    st.stop()

df_transition = run_transition_demo(selected_scenarios)
brsr = run_brsr_demo(renewable_share, verified_data, target_coverage=50)
integrated = build_integrated(df_transition, brsr["signal"])
report = integrated["report_worst"]
peak = integrated["peak"]

# ============================================================
# TABS
# ============================================================
tab_overview, tab_transition, tab_brsr, tab_integrated, tab_board, tab_access = st.tabs([
    "🏠 Overview", "⚡ Transition", "📘 BRSR", "🧩 Integrated", "🤖 Board Summary", "🚀 Request Access"
])

with tab_overview:
    st.subheader("Selected demo outputs")
    st.caption("Only the most important cards are shown in the public demo. Full pilot access includes detailed diagnostics and exports.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(f"{DEMO['reporting_year']} worst-scenario PD", fmt_pct(report["PD"]), "Transition risk, reporting-year view")
    with c2:
        metric_card(f"{DEMO['reporting_year']} ECL", fmt_money(report["ECL"]), "Expected credit loss in ₹ Cr", "#FFD166")
    with c3:
        metric_card("Capital signal", integrated["capital_signal"], "Board-level stress indicator", "#FF6B6B" if integrated["capital_signal"] in ["Severe", "Elevated"] else "#80ED99")
    with c4:
        metric_card("BRSR signal", f"+{brsr['signal']*10000:.0f} bps", "Operational climate governance signal", "#A78BFA")

    st.markdown("---")
    a, b = st.columns([2, 1])
    with a:
        st.plotly_chart(plot_line(df_transition, "PD", "PD trajectory — limited scenario view", tickformat=".1%"), width="stretch")
    with b:
        st.markdown("### What full access adds")
        locked_card("Full financial decomposition", "Revenue, EBITDA, DSCR, LGD, ECL drivers and audit-ready tables.")
        locked_card("Physical GIS risk", "Asset-level flood, heat and cyclone analysis with exposure allocation.")

with tab_transition:
    st.subheader("Transition risk — limited public view")
    st.caption("Shown: PD, ECL, and DSCR trends. Hidden: proprietary equations, calibration parameters, detailed intermediate tables.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Worst reporting-year DSCR", fmt_num(report["DSCR"], "×"), "Lower values imply weaker debt-service capacity", "#FF6B6B" if report["DSCR"] < 1.2 else "#80ED99")
    with c2:
        metric_card("Carbon burden", fmt_pct(report["Carbon_Burden"]), "Net carbon cost as % of revenue", "#FFD166")
    with c3:
        non_viable = "Yes" if integrated["first_non_viable"] else "Not by 2040"
        metric_card("Business-model stress", non_viable, "Full app shows non-viable year and drivers", "#FF6B6B" if integrated["first_non_viable"] else "#80ED99")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_line(df_transition, "ECL", "ECL trajectory — ₹ Cr", money=True), width="stretch")
    with col2:
        st.plotly_chart(plot_line(df_transition, "DSCR", "DSCR stress trajectory"), width="stretch")

    st.markdown('<div class="success-note">Full pilot access includes editable financial inputs, sector calibration, full scenario tables, target planning and downloadable board reports.</div>', unsafe_allow_html=True)

with tab_brsr:
    st.subheader("BRSR diagnostics — limited public view")
    st.caption("The demo exposes only a small number of BRSR levers. The full tool includes deeper BRSR Core readiness and operational-risk mapping.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("BRSR readiness", f"{brsr['readiness']:.0f}%", "Limited readiness score", "#80ED99" if brsr["readiness"] >= 60 else "#FFD166")
    with c2:
        metric_card("Governance signal", f"+{brsr['signal']*10000:.0f} bps", "Converted to bounded risk factor", "#A78BFA")
    with c3:
        metric_card("Open flags", str(len(brsr["flags"])), "Detailed flag weights hidden", "#FF6B6B" if brsr["flags"] else "#80ED99")

    if brsr["flags"]:
        st.markdown("**Visible flags in this limited demo:**")
        for flag in brsr["flags"]:
            st.markdown(f"- {flag}")
    else:
        st.success("No visible BRSR flags in the selected demo configuration.")

    locked_card("Advanced BRSR Core module", "Full version includes GHG intensity benchmarking, water/waste risk, readiness radar, target planning and full governance notes.")

with tab_integrated:
    st.subheader("Integrated climate-credit output")
    st.caption("Public demo shows the board-level result only. Advanced integration and model governance are hidden.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Integrated PD", fmt_pct(integrated["integrated_pd"]), "Bounded scenario + governance result")
    with c2:
        metric_card("Integrated ECL", fmt_money(integrated["integrated_ecl"]), "₹ Cr; reporting-year stress", "#FFD166")
    with c3:
        metric_card("Risk score", f"{integrated['risk_score']}/100", "Board-level indicator", "#FF6B6B" if integrated["risk_score"] >= 60 else "#80ED99")
    with c4:
        metric_card("BRSR factor", f"×{integrated['brsr_mult']:.3f}", "Exact method hidden in demo", "#A78BFA")

    st.markdown("### Locked in full access")
    l1, l2, l3 = st.columns(3)
    with l1:
        locked_card("Integrated decomposition")
    with l2:
        locked_card("Monte Carlo climate stress")
    with l3:
        locked_card("ICAAP / board export")

with tab_board:
    st.subheader("AI-style board summary — fixed public version")
    st.caption("This public version uses a concise fixed narrative to avoid exposing prompts or sending user data to external AI services.")
    st.markdown(f"""
    <div class="success-note">
    <b>Board summary:</b><br><br>
    For the fictional steel borrower, transition stress under the selected scenario set creates a material credit-risk signal in the {DEMO['reporting_year']} reporting year. The worst-scenario PD reaches <b>{fmt_pct(report['PD'])}</b>, with estimated ECL of <b>{fmt_money(report['ECL'])}</b>. The integrated capital signal is <b>{integrated['capital_signal']}</b>.<br><br>
    <b>Priority actions:</b><br>
    1. Review debt-service resilience under high carbon-price scenarios.<br>
    2. Improve BRSR readiness by increasing renewable energy share and verified disclosure coverage.<br>
    3. Request full pilot access for physical GIS risk, calibration, and board-report export.
    </div>
    """, unsafe_allow_html=True)
    locked_card("Live CFO / Risk / Strategy advisor", "Available in the full product with token-safe AI and internal governance controls.")

with tab_access:
    st.subheader("Request full pilot access")
    st.markdown("""
    The public demo intentionally hides advanced functionality. Serious users can request pilot access for:

    - Custom company inputs and sector selection
    - Physical risk module with asset-level GIS
    - Full transition, BRSR and integrated-risk tables
    - Monte Carlo climate stress
    - Calibration and validation workspace
    - Excel / board-report export
    - Private deployment or white-label version
    """)
    st.markdown(f"""
    <div class="demo-card">
      <div class="card-label">Contact</div>
      <div class="card-value" style="font-size:22px;">{CONTACT_EMAIL}</div>
      <div class="card-note">Mention: ICCRE LinkedIn Demo · {st.session_state.lead.get('organisation','')}</div>
    </div>
    """, unsafe_allow_html=True)

    summary_payload = {
        "company": DEMO["company_name"],
        "reporting_year": DEMO["reporting_year"],
        "worst_reporting_year_pd": f"{report['PD']:.4f}",
        "worst_reporting_year_ecl_cr": f"{report['ECL']:.2f}",
        "capital_signal": integrated["capital_signal"],
        "demo_user_org": st.session_state.lead.get("organisation", ""),
    }
    st.download_button(
        "Download limited demo summary",
        data=json.dumps(summary_payload, indent=2).encode("utf-8"),
        file_name="ICCRE_limited_demo_summary.json",
        mime="application/json",
        width="stretch",
    )

st.markdown("---")
st.caption("© ICCRE demo. Fictional data. Scenario-analysis only. Not financial, credit, investment, regulatory, or legal advice. Do not enter confidential company data in this public demo.")
