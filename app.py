"""
PharmaCast - Pharma sales forecasting & inventory planning dashboard.
Run locally:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))
from pipeline import CATEGORIES, load_weekly  # noqa: E402

st.set_page_config(page_title="PharmaCast | Pharma Sales Forecasting", page_icon="💊", layout="wide")

# Fixed colour per model (identity never changes with filters)
MODEL_COLORS = {
    "Seasonal naive": "#9c9c9c",
    "Moving average (4w)": "#f2a600",
    "Holt-Winters": "#00b894",
    "LightGBM": "#6a4ce2",
    "Ensemble (HW + LightGBM)": "#257de0",
}
ACTUAL_COLOR = "#191919"
CAT_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
BASELINE = "Seasonal naive"
BEST = "Ensemble (HW + LightGBM)"


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.18),
                      font=dict(color="#191919"))
    fig.update_xaxes(showgrid=False, linecolor="#cccccc")
    fig.update_yaxes(gridcolor="#eeeeee", zeroline=False)
    return fig


REQUIRED_DATA = [ROOT / "data" / "salesweekly.csv", ROOT / "data" / "salesdaily.csv"]
REQUIRED_OUT = [OUT / f for f in ("metrics.csv", "backtest.csv", "future_forecast.csv", "summary.json")]

missing_data = [str(p.relative_to(ROOT)) for p in REQUIRED_DATA if not p.exists()]
if missing_data:
    st.error("These data files are missing from the repository: " + ", ".join(missing_data)
             + ". Upload the `data` folder to GitHub and reboot the app.")
    st.stop()

if any(not p.exists() for p in REQUIRED_OUT):  # outputs not uploaded: build them once, in-process
    with st.spinner("First run: training the models (about 1 minute)..."):
        import pipeline
        pipeline.main()


@st.cache_data
def load_all():
    wide = load_weekly()
    daily = pd.read_csv(ROOT / "data" / "salesdaily.csv")
    daily["datum"] = pd.to_datetime(daily["datum"], format="%m/%d/%Y")
    metrics = pd.read_csv(OUT / "metrics.csv")
    bt = pd.read_csv(OUT / "backtest.csv", parse_dates=["origin", "week"])
    fut = pd.read_csv(OUT / "future_forecast.csv", parse_dates=["week"])
    summary = json.loads((OUT / "summary.json").read_text())
    return wide, daily, metrics, bt, fut, summary


wide, daily, metrics, bt, fut, summary = load_all()
overall = metrics[metrics.category == "ALL"].set_index("model")
long = wide.melt(id_vars="week", var_name="category", value_name="sales")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("💊 PharmaCast")
    st.caption("Weekly drug demand forecasting for inventory and supply planning")
    st.markdown(
        f"**Data:** {summary['n_weeks']} weeks of real pharmacy point-of-sale data, "
        f"{summary['data_start']} to {summary['data_end']}, 8 ATC drug categories."
    )
    st.markdown("**Source:** [Kaggle: Pharma sales data](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)")
    st.markdown("**Stack:** Python, pandas, statsmodels, LightGBM, Plotly, Streamlit (all open source)")
    st.divider()
    st.caption("Built by Sagar as an end-to-end portfolio project")

st.title("Pharma Sales Forecasting & Inventory Planner")
st.markdown(
    "Forecast weekly demand for 8 drug categories **4 weeks ahead** (a typical replenishment lead time), "
    "compare simple baselines with statistical and ML models, and turn the forecast into "
    "**safety stock and reorder decisions**."
)

tab_over, tab_eda, tab_models, tab_plan, tab_method = st.tabs(
    ["📌 Overview", "🔎 EDA", "📈 Model comparison", "📦 Forecast & inventory planner", "🧠 Method & business impact"]
)

# ------------------------------------------------------------------ overview
with tab_over:
    b, m = overall.loc[BASELINE], overall.loc[BEST]
    wape_gain = (b["WAPE %"] - m["WAPE %"]) / b["WAPE %"] * 100
    rmse_gain = (b["RMSE"] - m["RMSE"]) / b["RMSE"] * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Forecast error (WAPE), best model", f"{m['WAPE %']:.1f}%", f"-{b['WAPE %'] - m['WAPE %']:.1f} pts vs baseline", delta_color="inverse")
    c2.metric("Error reduction vs seasonal naive", f"{wape_gain:.0f}%")
    c3.metric("RMSE, best model", f"{m['RMSE']:.1f} units", f"-{rmse_gain:.0f}% vs baseline", delta_color="inverse")
    c4.metric("Backtest window (13 rolling forecasts)", f"{summary['test_weeks']} weeks")

    st.subheader("What this project answers")
    st.markdown(
        """
- **How much of each drug category will we sell in the next 4 to 12 weeks?**
- **Which forecasting approach is accurate enough to plan with?** Simple rules (same week last year, recent average) vs Holt-Winters vs LightGBM vs an ensemble.
- **What does better accuracy buy the business?** Less safety stock for the same service level, fewer stock-outs of fast movers, and better-timed field and supply effort around seasonal peaks (e.g. allergy and respiratory drugs in spring).
"""
    )
    st.subheader("Total weekly sales by category")
    fig = px.area(long, x="week", y="sales", color="category", color_discrete_sequence=CAT_COLORS,
                  labels={"sales": "Units sold per week", "week": ""})
    fig.update_traces(line=dict(width=0.5))
    st.plotly_chart(style(fig, 420), width="stretch")

# ------------------------------------------------------------------ EDA
with tab_eda:
    st.subheader("1. How big is each category?")
    share = long.groupby("category")["sales"].sum().sort_values(ascending=True).reset_index()
    share["share %"] = share["sales"] / share["sales"].sum() * 100
    share["description"] = share["category"].map(CATEGORIES)
    fig = px.bar(share, x="sales", y="category", orientation="h", hover_data=["description", "share %"],
                 labels={"sales": "Total units 2014 to 2019", "category": ""}, color_discrete_sequence=["#257de0"])
    fig.update_traces(hovertemplate="%{y}: %{x:,.0f} units<br>%{customdata[0]}<extra></extra>")
    st.plotly_chart(style(fig, 330).update_layout(hovermode="closest"), width="stretch")
    st.caption("N02BE (paracetamol-type analgesics) is about 40% of volume, so errors there matter most for stock value. "
               "N05C (sedatives) sells only ~4 units a week, so percentage errors there look large but are tiny in units.")

    st.subheader("2. Trend and seasonality for one category")
    cat = st.selectbox("Category", list(CATEGORIES), index=7, format_func=lambda c: f"{c} - {CATEGORIES[c]}", key="eda_cat")
    s = wide.set_index("week")[cat]
    fig = go.Figure()
    fig.add_scatter(x=s.index, y=s, name="Weekly sales", line=dict(color="#84b9f3", width=1.5))
    fig.add_scatter(x=s.index, y=s.rolling(13, center=True).mean(), name="13-week average (trend)", line=dict(color="#1757ac", width=2.5))
    st.plotly_chart(style(fig).update_layout(yaxis_title="Units per week"), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Seasonality: average sales index by month (1.0 = normal month)**")
        mon = long.assign(month=long.week.dt.month).groupby(["category", "month"])["sales"].mean().unstack()
        idx = mon.div(mon.mean(axis=1), axis=0)
        fig = px.imshow(idx.round(2), color_continuous_scale=["#f47230", "#f4f4f4", "#257de0"], zmin=0.4, zmax=1.6,
                        aspect="auto", labels=dict(x="Month", y="", color="Index"), text_auto=".1f")
        fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=list("JFMAMJJASOND"))
        st.plotly_chart(style(fig, 360).update_layout(hovermode="closest"), width="stretch")
        st.caption("R06 (antihistamines) peaks in spring/early summer; R03 (respiratory) and N02BA rise in colder months. "
                   "This is why a seasonal model is needed.")
    with col2:
        st.markdown("**Day-of-week pattern (daily data)**")
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow = daily.assign(total=daily[list(CATEGORIES)].sum(axis=1)).groupby("Weekday Name")["total"].mean().reindex(order).reset_index()
        fig = px.bar(dow, x="Weekday Name", y="total", labels={"total": "Avg units per day", "Weekday Name": ""},
                     color_discrete_sequence=["#257de0"])
        st.plotly_chart(style(fig, 360).update_layout(hovermode="closest"), width="stretch")
        st.caption("Sunday is quieter. Daily data also shows days with zero sales (pharmacy closed on public holidays such as "
                   "1 Jan and 7 Jan), which pull some weekly totals down.")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Yearly totals by category (growth/decline)**")
        yr = long[long.week.dt.year < 2019].assign(year=lambda d: d.week.dt.year).groupby(["year", "category"])["sales"].sum().reset_index()
        yr["index"] = yr.groupby("category")["sales"].transform(lambda x: x / x.iloc[0] * 100)
        fig = px.line(yr, x="year", y="index", color="category", markers=True, color_discrete_sequence=CAT_COLORS,
                      labels={"index": "Index (2014 = 100)", "year": ""})
        st.plotly_chart(style(fig, 360), width="stretch")
        st.caption("Full years only (2019 is partial). Indexed to 2014 so categories of different size can share one axis.")
    with col4:
        st.markdown("**How categories move together (correlation of weekly sales)**")
        corr = wide[list(CATEGORIES)].corr().round(2)
        fig = px.imshow(corr, color_continuous_scale=["#ec4647", "#f4f4f4", "#257de0"], zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        st.plotly_chart(style(fig, 360).update_layout(hovermode="closest"), width="stretch")
        st.caption("Positive correlations mean shared drivers (overall footfall). That is why one global LightGBM model "
                   "trained across all categories can learn shared patterns.")

    with st.expander("Data quality checks"):
        st.markdown(
            f"""
- {len(wide)} complete weeks after dropping the final week (it contained only 2 days of data).
- No missing values. {int((wide[list(CATEGORIES)] == 0).sum().sum())} zero weeks, all in N05C (a slow mover), so MAPE skips zero weeks and **WAPE** is reported as the main business metric.
- Units can be fractional because the source records packs split into smaller doses.
"""
        )
        st.dataframe(wide.describe().T.round(1), width="stretch")

# ------------------------------------------------------------------ models
with tab_models:
    st.subheader("Backtest results: last 52 weeks, forecasting 4 weeks ahead, retrained every 4 weeks")
    tbl = overall[["MAPE %", "WAPE %", "RMSE"]].sort_values("WAPE %")
    tbl["WAPE gain vs baseline"] = (1 - tbl["WAPE %"] / overall.loc[BASELINE, "WAPE %"]) * 100
    st.dataframe(tbl.style.format({"MAPE %": "{:.1f}", "WAPE %": "{:.1f}", "RMSE": "{:.1f}", "WAPE gain vs baseline": "{:+.1f}%"})
                 .highlight_min(subset=["WAPE %", "RMSE"], color="#cce3fd"), width="stretch")
    st.caption("WAPE = total absolute error / total actual sales. It is volume-weighted, so it reflects money and stock better than MAPE, "
               "which is inflated by tiny categories. Lower is better for all three.")

    per = metrics[metrics.category != "ALL"]
    metric = st.radio("Metric by category", ["WAPE %", "MAPE %", "RMSE"], horizontal=True)
    fig = px.bar(per, x="category", y=metric, color="model", barmode="group", color_discrete_map=MODEL_COLORS,
                 category_orders={"model": list(MODEL_COLORS)})
    st.plotly_chart(style(fig, 400).update_layout(hovermode="closest"), width="stretch")

    st.subheader("Actual vs forecast in the backtest")
    c1, c2 = st.columns([1, 2])
    cat2 = c1.selectbox("Category", list(CATEGORIES), index=3, format_func=lambda c: f"{c} - {CATEGORIES[c]}", key="bt_cat")
    models_sel = c2.multiselect("Models", list(MODEL_COLORS), default=[BASELINE, BEST])
    sub = bt[bt.category == cat2]
    hist = wide.set_index("week")[cat2]
    hist = hist[hist.index >= sub.week.min() - pd.Timedelta(weeks=26)]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist, name="Actual", line=dict(color=ACTUAL_COLOR, width=2))
    for mdl in models_sel:
        d = sub[sub.model == mdl]
        fig.add_scatter(x=d.week, y=d.forecast, name=mdl, line=dict(color=MODEL_COLORS[mdl], width=2, dash="dot" if mdl == BASELINE else "solid"))
    fig.add_vline(x=sub.week.min(), line_dash="dash", line_color="#9c9c9c")
    fig.add_annotation(x=sub.week.min(), y=1, yref="paper", text="backtest starts", showarrow=False, xanchor="left", font=dict(color="#666666"))
    st.plotly_chart(style(fig, 420).update_layout(yaxis_title="Units per week"), width="stretch")

# ------------------------------------------------------------------ planner
with tab_plan:
    st.subheader("Turn the forecast into an order plan")
    c1, c2, c3, c4 = st.columns(4)
    cat3 = c1.selectbox("Category", list(CATEGORIES), index=3, format_func=lambda c: f"{c} - {CATEGORIES[c]}", key="plan_cat")
    service = c2.slider("Target service level (chance of no stock-out)", 0.80, 0.99, 0.95, 0.01)
    lead = c3.slider("Supplier lead time (weeks)", 1, 4, 2)
    cost = c4.number_input("Avg cost per unit (₹)", min_value=1.0, value=120.0, step=10.0)
    model_choice = st.radio("Forecast model", list(MODEL_COLORS), index=list(MODEL_COLORS).index(BEST), horizontal=True)

    f = fut[(fut.category == cat3) & (fut.model == model_choice)].sort_values("week")
    sd = f.resid_sd.iloc[0]
    z = norm.ppf(service)
    hist = wide.set_index("week")[cat3].iloc[-52:]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist, name="Actual (last 52 weeks)", line=dict(color=ACTUAL_COLOR, width=2))
    upper = f.forecast + z * sd
    lower = (f.forecast - z * sd).clip(lower=0)
    fig.add_scatter(x=list(f.week) + list(f.week[::-1]), y=list(upper) + list(lower[::-1]), fill="toself",
                    fillcolor="rgba(37,125,224,0.15)", line=dict(width=0), name=f"{service:.0%} range", hoverinfo="skip")
    fig.add_scatter(x=f.week, y=f.forecast, name=f"Forecast: {model_choice}", line=dict(color=MODEL_COLORS[model_choice], width=2.5))
    st.plotly_chart(style(fig, 420).update_layout(yaxis_title="Units per week"), width="stretch")
    st.caption("The shaded range uses the model's real backtest error, not a theoretical assumption. "
               "The first 4 weeks are the most reliable; weeks 5 to 12 are for capacity planning.")

    # Safety stock = z * sigma_error * sqrt(lead time)
    def safety_stock(mdl):
        s_ = fut[(fut.category == cat3) & (fut.model == mdl)].resid_sd.iloc[0]
        return z * s_ * np.sqrt(lead)

    ss_model, ss_base = safety_stock(model_choice), safety_stock(BASELINE)
    lead_demand = f.forecast.iloc[:lead].sum()
    rop = lead_demand + ss_model
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Demand over next {lead} wk", f"{lead_demand:,.0f} units")
    k2.metric("Safety stock needed", f"{ss_model:,.0f} units", f"{ss_model - ss_base:+,.0f} vs baseline", delta_color="inverse")
    k3.metric("Reorder point", f"{rop:,.0f} units", help="Order more when stock on hand falls below this level")
    k4.metric("Working capital in safety stock", f"₹{ss_model * cost:,.0f}", f"₹{(ss_model - ss_base) * cost:+,.0f} vs baseline", delta_color="inverse")

    st.markdown("**All categories: safety stock with the best model vs the seasonal-naive baseline**")
    rows = []
    for c in CATEGORIES:
        e_b = fut[(fut.category == c) & (fut.model == BASELINE)].resid_sd.iloc[0]
        e_m = fut[(fut.category == c) & (fut.model == BEST)].resid_sd.iloc[0]
        rows.append(dict(category=c, description=CATEGORIES[c],
                         **{"next 4 wk demand": fut[(fut.category == c) & (fut.model == BEST)].forecast.iloc[:4].sum(),
                            "safety stock (baseline)": z * e_b * np.sqrt(lead),
                            "safety stock (ensemble)": z * e_m * np.sqrt(lead)}))
    ss = pd.DataFrame(rows)
    ss["units saved"] = ss["safety stock (baseline)"] - ss["safety stock (ensemble)"]
    ss["₹ freed"] = ss["units saved"] * cost
    total = ss[["safety stock (baseline)", "safety stock (ensemble)", "units saved", "₹ freed"]].sum()
    st.dataframe(ss.style.format({c: "{:,.0f}" for c in ss.columns if c not in ("category", "description")}), width="stretch", hide_index=True)
    st.success(
        f"At a {service:.0%} service level and {lead}-week lead time, the ensemble needs "
        f"**{total['safety stock (ensemble)']:,.0f}** units of safety stock vs **{total['safety stock (baseline)']:,.0f}** with the baseline: "
        f"**{(1 - total['safety stock (ensemble)'] / total['safety stock (baseline)']) * 100:.0f}% less stock** for the same protection "
        f"(≈ ₹{total['₹ freed']:,.0f} working capital per pharmacy at ₹{cost:,.0f}/unit). Across a chain or distributor network this scales with the number of outlets."
    )
    st.download_button("Download 12-week forecast (CSV)", fut[fut.model == model_choice].to_csv(index=False),
                       file_name=f"forecast_{model_choice.split()[0].lower()}.csv", mime="text/csv")

# ------------------------------------------------------------------ method
with tab_method:
    st.subheader("How it works")
    st.markdown(
        f"""
**1. Problem framing.** A pharmacy, distributor or pharma supply team must order stock before demand happens. Too little means stock-outs and lost patients; too much means expired stock and locked-up cash. The decision needs a forecast about **{summary['horizon_weeks']} weeks ahead**.

**2. Data.** Real point-of-sale data from one pharmacy (about 600,000 transactions, 2014 to 2019), grouped into 8 ATC (Anatomical Therapeutic Chemical) drug categories and aggregated to weekly totals.

**3. Models.**
| Model | Idea | Why include it |
|---|---|---|
| Seasonal naive | Next week = same week last year | Standard baseline: any model must beat this |
| Moving average (4w) | Next weeks = average of last 4 weeks | Second baseline, captures recent level |
| Holt-Winters | Exponential smoothing with damped trend and 52-week seasonality | Classic, explainable statistical model, one per category |
| LightGBM | Gradient boosted trees on lag, rolling and calendar features, one global model for all categories | Learns non-linear and shared patterns; features only use data ≥ 4 weeks old, so no leakage |
| Ensemble | Average of Holt-Winters and LightGBM | Their errors are different, so averaging cancels part of them |

**4. Validation.** Rolling-origin backtest: 13 times over the last 52 weeks, train on everything before the cut-off, forecast the next 4 weeks, then move forward 4 weeks. This mimics how the model would be used in real life.

**5. Metrics.** MAPE (easy to explain), WAPE (volume weighted, best for business), RMSE (punishes big misses, which cause stock-outs).

**6. From forecast to decision.** Safety stock = z × σ(forecast error) × √(lead time). A model with lower error needs less buffer stock for the same service level. That is the money story.
"""
        )
    st.subheader("Business impact")
    st.markdown(
        """
- **Inventory:** lower forecast error → smaller safety stock → less working capital and less expiry write-off, with the same or better availability.
- **Supply planning:** a 12-week view lets the supply team pre-book production or imports before seasonal peaks (antihistamines in spring, respiratory drugs in winter).
- **Sales rep allocation:** the same category-level forecast tells commercial teams where demand is rising, so field effort and samples can shift to allergy products before the season, not after.
- **Limitations and next steps:** one pharmacy only; no price, promotion, weather or epidemiology data. Next steps: add holiday and weather features, forecast per outlet or SKU (hierarchical reconciliation), and monitor drift monthly.
"""
    )
