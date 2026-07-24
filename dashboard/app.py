"""Streamlit multi-page dashboard for customer churn prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)


# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def api_health() -> dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False}


@st.cache_data(ttl=300)
def api_model_info() -> dict[str, Any]:
    try:
        resp = requests.get(f"{API_BASE}/model/info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def api_predict(features: dict[str, float], explain: bool = False) -> dict[str, Any] | None:
    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json={"features": features, "explain": explain},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("🔮 Churn Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Data Overview", "Feature Analysis", "Model Performance", "What-if Simulator", "Batch Predict"],
)

st.sidebar.markdown("---")
health = api_health()
if health.get("status") == "ok":
    st.sidebar.success("✅ Backend connected")
else:
    st.sidebar.error("❌ Backend unreachable")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: Data Overview
# ═══════════════════════════════════════════════════════════════════════════
def page_data_overview() -> None:
    st.header("📋 Data Overview")
    st.markdown("Exploratory analysis of the Cell2Cell customer churn dataset.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset", "Cell2Cell")
    col2.metric("Rows", "~71,000")
    col3.metric("Features", "~70 (incl. engineered)")

    st.subheader("Target Distribution")
    img_path = REPORTS_DIR / "churn_distribution.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Run the EDA notebook to generate target distribution plots.")

    st.subheader("Missing Values")
    img_miss = REPORTS_DIR / "missing_pct_bar.png"
    if img_miss.exists():
        st.image(str(img_miss), use_container_width=True)
    else:
        st.info("Run the EDA notebook to generate missing-values plots.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: Feature Analysis
# ═══════════════════════════════════════════════════════════════════════════
def page_feature_analysis() -> None:
    st.header("📈 Feature Analysis")
    st.markdown("Interactive exploration of feature distributions and relationships with churn.")

    report_images = list(REPORTS_DIR.glob("*.png")) if REPORTS_DIR.exists() else []

    if not report_images:
        st.info("No EDA plots found. Run the EDA notebook to generate them.")
        return

    image_names = [p.name for p in report_images]
    selected = st.selectbox("Select plot to view", sorted(image_names))

    if selected:
        st.image(str(REPORTS_DIR / selected), use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_path = REPORTS_DIR / "correlation_heatmap.png"
    if corr_path.exists():
        st.image(str(corr_path), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: Model Performance
# ═══════════════════════════════════════════════════════════════════════════
def page_model_performance() -> None:
    st.header("🏆 Model Performance")
    st.markdown("Trained model metrics and evaluation plots.")

    info = api_model_info()
    if info:
        metrics = info.get("metrics", {})
        if metrics:
            cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                cols[i].metric(k.replace("_", " ").title(), f"{v:.4f}")
        else:
            st.info("No metrics available. Train a model first.")
    else:
        st.info("Backend not available. Start the API to view model info.")

    st.subheader("ROC Curve")
    roc_path = REPORTS_DIR / "roc_curve.png"
    if roc_path.exists():
        st.image(str(roc_path), use_container_width=True)

    st.subheader("Precision-Recall Curve")
    pr_path = REPORTS_DIR / "pr_curve.png"
    if pr_path.exists():
        st.image(str(pr_path), use_container_width=True)

    st.subheader("Confusion Matrix")
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: What-if Simulator
# ═══════════════════════════════════════════════════════════════════════════
def page_what_if() -> None:
    st.header("🎯 What-if Simulator")
    st.markdown("Adjust features below to see how they affect churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        revenue = st.slider("Monthly Revenue", 0.0, 500.0, 100.0)
        mou = st.slider("Minutes of Use", 0.0, 3000.0, 500.0)
        months = st.slider("Tenure (months)", 0, 72, 12)
        overage = st.slider("Overage Charges", 0.0, 100.0, 10.0)
        dropvce = st.slider("Dropped Calls", 0, 50, 2)
        custcare = st.slider("Customer Care Calls", 0, 20, 1)

    with col2:
        age = st.slider("Age", 18, 80, 35)
        income = st.slider("Income Category", 1, 9, 4)
        eqpdays = st.slider("Days with Current Equipment", 0, 1500, 200)
        refurb = st.selectbox("Refurbished Phone", [0, 1])
        webcap = st.selectbox("Web Capable Phone", [0, 1])
        retcalls = st.slider("Retention Calls", 0, 10, 0)

    features = {
        "revenue": revenue,
        "mou": mou,
        "months": months,
        "overage": overage,
        "dropvce": dropvce,
        "custcare": custcare,
        "age1": age,
        "income": income,
        "eqpdays": eqpdays,
        "refurb": refurb,
        "webcap": webcap,
        "retcalls": retcalls,
    }

    if st.button("Predict Churn", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            result = api_predict(features, explain=True)

        if result:
            prob = result["churn_probability"]
            pred = result["prediction"]

            if pred == 1:
                st.error(f"⚠️ **Churn Predicted** — Probability: {prob:.1%}")
            else:
                st.success(f"✅ **No Churn** — Probability: {prob:.1%}")

            shap_vals = result.get("shap_values")
            if shap_vals:
                st.subheader("SHAP Feature Attribution")
                info = api_model_info()
                feat_names = info.get("feature_names", list(features.keys()))
                if len(shap_vals) == len(feat_names):
                    shap_df = pd.DataFrame(
                        {"feature": feat_names, "shap_value": shap_vals}
                    )
                    shap_df = shap_df[shap_df["feature"].isin(features.keys()) | (shap_df["shap_value"].abs() > 0.001)]
                    shap_df = shap_df.sort_values("shap_value", key=abs, ascending=False).head(12)
                else:
                    min_len = min(len(shap_vals), len(features))
                    shap_df = pd.DataFrame(
                        {"feature": list(features.keys())[:min_len], "shap_value": shap_vals[:min_len]}
                    ).sort_values("shap_value", key=abs, ascending=False)
                st.bar_chart(shap_df.set_index("feature"))



# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5: Batch Predict
# ═══════════════════════════════════════════════════════════════════════════
def page_batch_predict() -> None:
    st.header("📁 Batch Predict")
    st.markdown(
        "Upload a CSV file with customer features to get churn predictions for all rows. "
        "The CSV must contain all required feature columns."
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df)}** rows with **{len(df.columns)}** columns.")
        st.dataframe(df.head(10))

        if st.button("Run Predictions", type="primary"):
            with st.spinner("Sending batch to API..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict/batch", files=files, timeout=60
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")
                    return

            st.subheader("Results Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total", result["total_count"])
            col2.metric("Churned", result["churn_count"])
            col3.metric("Churn Rate", f"{result['churn_rate']:.1%}")

            results_df = pd.DataFrame(
                [
                    {"index": p.get("index", i), "prediction": p["prediction"],
                     "churn_probability": p["churn_probability"]}
                    for i, p in enumerate(result.get("predictions", result.get("results", [])))
                ]
            )
            st.dataframe(results_df)

            csv_output = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results CSV",
                data=csv_output,
                file_name="predictions.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════
PAGES = {
    "Data Overview": page_data_overview,
    "Feature Analysis": page_feature_analysis,
    "Model Performance": page_model_performance,
    "What-if Simulator": page_what_if,
    "Batch Predict": page_batch_predict,
}

PAGES[page]()

st.sidebar.markdown("---")
st.sidebar.caption("Churn Prediction v0.1.0")
