"""
Pharma Sales Forecasting pipeline.

Steps
1. Load weekly pharmacy sales by ATC drug category (public Kaggle dataset).
2. Clean (drop the last, incomplete week).
3. Rolling-origin backtest over the last 52 weeks, forecasting 4 weeks ahead
   each time (a typical replenishment lead time):
     - Baseline 1: Seasonal naive (same week last year)
     - Baseline 2: 4-week moving average
     - Model 3:   Holt-Winters exponential smoothing (per category)
     - Model 4:   LightGBM global model (all categories, lag + calendar features)
     - Model 5:   Ensemble = average of Holt-Winters and LightGBM
4. Score with MAPE, WAPE and RMSE.
5. Fit the best model on all data and produce a 12-week forecast.
6. Save everything to outputs/ so the Streamlit app loads instantly.

Run:  python src/pipeline.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"

CATEGORIES = {
    "M01AB": "Anti-inflammatory, acetic acid derivatives (e.g. diclofenac)",
    "M01AE": "Anti-inflammatory, propionic acid derivatives (e.g. ibuprofen)",
    "N02BA": "Analgesics, salicylic acid (e.g. aspirin)",
    "N02BE": "Analgesics, pyrazolones and anilides (e.g. paracetamol)",
    "N05B": "Anxiolytics (e.g. diazepam)",
    "N05C": "Hypnotics and sedatives",
    "R03": "Drugs for obstructive airway diseases (e.g. asthma inhalers)",
    "R06": "Antihistamines for systemic use (allergy)",
}

HORIZON = 4          # weeks ahead we need to know demand (replenishment lead time)
TEST_WEEKS = 52      # backtest window
FUTURE_WEEKS = 12    # forward forecast shown in the app
SEASON = 52


# --------------------------------------------------------------------------- data
def load_weekly() -> pd.DataFrame:
    df = pd.read_csv(DATA / "salesweekly.csv")
    df["datum"] = pd.to_datetime(df["datum"], format="%m/%d/%Y")
    df = df.rename(columns={"datum": "week"}).sort_values("week")
    # The final row only contains 2 days of data (7-8 Oct 2019), so drop it.
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def to_long(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.melt(id_vars="week", var_name="category", value_name="sales")


# --------------------------------------------------------------------------- metrics
def mape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mask = y > 0  # MAPE is undefined when actual = 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


# --------------------------------------------------------------------------- models
def seasonal_naive(train: pd.Series, h: int) -> np.ndarray:
    return train.iloc[-SEASON: -SEASON + h].to_numpy() if h < SEASON else train.iloc[-SEASON:].to_numpy()[:h]


def moving_average(train: pd.Series, h: int, window: int = 4) -> np.ndarray:
    return np.repeat(train.iloc[-window:].mean(), h)


def holt_winters(train: pd.Series, h: int) -> np.ndarray:
    model = ExponentialSmoothing(
        train.clip(lower=0.1).to_numpy(),
        trend="add", damped_trend=True,
        seasonal="add", seasonal_periods=SEASON,
        initialization_method="estimated",
    ).fit(optimized=True)
    return np.clip(model.forecast(h), 0, None)


def make_features(long: pd.DataFrame) -> pd.DataFrame:
    """Features use only information available HORIZON weeks earlier,
    so one model can forecast 1..HORIZON weeks ahead without leakage."""
    f = long.sort_values(["category", "week"]).copy()
    g = f.groupby("category")["sales"]
    for lag in range(HORIZON, HORIZON + 5):
        f[f"lag_{lag}"] = g.shift(lag)
    f["lag_52"] = g.shift(52)
    f["lag_51"] = g.shift(51)
    f["lag_53"] = g.shift(53)
    shifted = g.shift(HORIZON)
    f["roll_mean_4"] = shifted.groupby(f["category"]).transform(lambda s: s.rolling(4).mean())
    f["roll_mean_13"] = shifted.groupby(f["category"]).transform(lambda s: s.rolling(13).mean())
    f["roll_std_13"] = shifted.groupby(f["category"]).transform(lambda s: s.rolling(13).std())
    f["weekofyear"] = f["week"].dt.isocalendar().week.astype(int)
    f["month"] = f["week"].dt.month
    f["year"] = f["week"].dt.year
    f["cat_code"] = f["category"].astype("category").cat.codes
    # scale-free target: ratio to recent level helps one model serve all categories
    f["level"] = f["roll_mean_13"]
    return f


FEATURES = [f"lag_{l}" for l in range(HORIZON, HORIZON + 5)] + [
    "lag_51", "lag_52", "lag_53", "roll_mean_4", "roll_mean_13", "roll_std_13",
    "weekofyear", "month", "cat_code",
]
LGB_PARAMS = dict(
    objective="regression", learning_rate=0.03, n_estimators=600, num_leaves=15,
    min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, verbose=-1, random_state=42,
)


def lgb_fit(train_feat: pd.DataFrame) -> lgb.LGBMRegressor:
    tr = train_feat.dropna(subset=FEATURES + ["sales"])
    # predict sales relative to recent level -> categories of different size share patterns
    y = tr["sales"] / tr["level"].clip(lower=0.5)
    X = tr[FEATURES].copy()
    for c in [c for c in FEATURES if c.startswith(("lag_", "roll_"))]:
        X[c] = X[c] / tr["level"].clip(lower=0.5)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model


def lgb_predict(model, feat: pd.DataFrame) -> np.ndarray:
    X = feat[FEATURES].copy()
    lvl = feat["level"].clip(lower=0.5)
    for c in [c for c in FEATURES if c.startswith(("lag_", "roll_"))]:
        X[c] = X[c] / lvl
    return np.clip(model.predict(X) * lvl.to_numpy(), 0, None)


# --------------------------------------------------------------------------- backtest
def backtest(wide: pd.DataFrame) -> pd.DataFrame:
    long = to_long(wide)
    feat = make_features(long)
    weeks = wide["week"].to_numpy()
    n = len(weeks)
    origins = list(range(n - TEST_WEEKS, n, HORIZON))  # index of first forecast week
    rows = []
    for o in origins:
        h = min(HORIZON, n - o)
        train_weeks, test_weeks = weeks[:o], weeks[o:o + h]
        # LightGBM global model, retrained at each origin (no peeking at the future)
        model = lgb_fit(feat[feat["week"].isin(train_weeks)])
        test_feat = feat[feat["week"].isin(test_weeks)]
        lgb_pred = dict(zip(zip(test_feat["category"], test_feat["week"]), lgb_predict(model, test_feat)))
        for cat in CATEGORIES:
            s_train = wide.set_index("week")[cat].loc[train_weeks]
            actual = wide.set_index("week")[cat].loc[test_weeks].to_numpy()
            preds = {
                "Seasonal naive": seasonal_naive(s_train, h),
                "Moving average (4w)": moving_average(s_train, h),
                "Holt-Winters": holt_winters(s_train, h),
                "LightGBM": np.array([lgb_pred[(cat, w)] for w in test_weeks]),
            }
            for m, p in preds.items():
                for i, w in enumerate(test_weeks):
                    rows.append(dict(origin=weeks[o], week=w, step=i + 1, category=cat,
                                     model=m, actual=actual[i], forecast=float(p[i])))
    return add_ensemble(pd.DataFrame(rows), ["origin", "week", "step", "category"])


ENSEMBLE = "Ensemble (HW + LightGBM)"


def add_ensemble(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average of the two strongest, very different models (statistical + ML)."""
    parts = df[df["model"].isin(["Holt-Winters", "LightGBM"])]
    agg = {"forecast": "mean"}
    if "actual" in df:
        agg["actual"] = "first"
    ens = parts.groupby(keys, as_index=False).agg(agg)
    ens["model"] = ENSEMBLE
    return pd.concat([df, ens[df.columns]], ignore_index=True)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        return pd.Series({"MAPE %": mape(g.actual, g.forecast),
                          "WAPE %": wape(g.actual, g.forecast),
                          "RMSE": rmse(g.actual, g.forecast)})
    by_cat = bt.groupby(["category", "model"]).apply(agg, include_groups=False).reset_index()
    overall = bt.groupby("model").apply(agg, include_groups=False).reset_index()
    overall["category"] = "ALL"
    return pd.concat([overall, by_cat], ignore_index=True)


# --------------------------------------------------------------------------- future forecast
def future_forecast(wide: pd.DataFrame, best_by_cat: dict[str, str], resid_sd: dict[str, float]) -> pd.DataFrame:
    """12-week forecast. LightGBM is applied recursively in blocks of HORIZON weeks."""
    last = wide["week"].max()
    future_weeks = pd.date_range(last + pd.Timedelta(weeks=1), periods=FUTURE_WEEKS, freq="7D")
    out = []
    # LightGBM recursive (block of 4 at a time, feeding forecasts back in)
    ext = to_long(wide)
    model = lgb_fit(make_features(ext))
    lgb_fc = {}
    for b in range(0, FUTURE_WEEKS, HORIZON):
        block = future_weeks[b:b + HORIZON]
        placeholder = pd.DataFrame([(w, c, np.nan) for w in block for c in CATEGORIES],
                                   columns=["week", "category", "sales"])
        tmp = pd.concat([ext, placeholder], ignore_index=True)
        f = make_features(tmp)
        f = f[f["week"].isin(block)]
        f["pred"] = lgb_predict(model, f)
        for _, r in f.iterrows():
            lgb_fc[(r.category, r.week)] = r.pred
        filled = f[["week", "category"]].assign(sales=f["pred"].to_numpy())
        ext = pd.concat([ext, filled], ignore_index=True)

    for cat in CATEGORIES:
        s = wide.set_index("week")[cat]
        fc = {
            "Seasonal naive": seasonal_naive(s, FUTURE_WEEKS),
            "Moving average (4w)": moving_average(s, FUTURE_WEEKS),
            "Holt-Winters": holt_winters(s, FUTURE_WEEKS),
            "LightGBM": np.array([lgb_fc[(cat, w)] for w in future_weeks]),
        }
        fc[ENSEMBLE] = (fc["Holt-Winters"] + fc["LightGBM"]) / 2
        for m, p in fc.items():
            for i, w in enumerate(future_weeks):
                out.append(dict(week=w, category=cat, model=m, forecast=float(p[i]),
                                is_best=(m == best_by_cat[cat]),
                                resid_sd=resid_sd[(cat, m)]))
    return pd.DataFrame(out)


def main():
    OUT.mkdir(exist_ok=True)
    wide = load_weekly()
    print(f"Weekly data: {wide.week.min().date()} -> {wide.week.max().date()}  ({len(wide)} weeks)")

    bt = backtest(wide)
    metrics = score(bt)
    bt.to_csv(OUT / "backtest.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    print("\nOverall backtest (last 52 weeks, 4 weeks ahead):")
    print(metrics[metrics.category == "ALL"].sort_values("WAPE %").round(2).to_string(index=False))

    # best model per category = lowest WAPE (robust when volumes are small)
    per_cat = metrics[metrics.category != "ALL"]
    best_by_cat = per_cat.loc[per_cat.groupby("category")["WAPE %"].idxmin()].set_index("category")["model"].to_dict()
    resid_sd = bt.assign(err=bt.actual - bt.forecast).groupby(["category", "model"])["err"].std().to_dict()

    fut = future_forecast(wide, best_by_cat, resid_sd)
    fut.to_csv(OUT / "future_forecast.csv", index=False)

    summary = {
        "data_start": str(wide.week.min().date()),
        "data_end": str(wide.week.max().date()),
        "n_weeks": int(len(wide)),
        "horizon_weeks": HORIZON,
        "test_weeks": TEST_WEEKS,
        "future_weeks": FUTURE_WEEKS,
        "best_model_by_category": best_by_cat,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nBest model per category:", best_by_cat)
    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
