"""
CarDekho Used Car Price Prediction
Streamlit Application — app.py

Run with:
    streamlit run app.py
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from data_preprocessing import preprocess_pipeline, engineer_features

MODEL_PATH = ROOT / "models" / "best_model.pkl"
DATA_DIR   = ROOT / "data"
CLEAN_CSV  = DATA_DIR / "cleaned_cars.csv"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CarDekho – Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2.4rem; }
    .main-header p  { color: #a8b2d8; margin: 0.4rem 0 0; font-size: 1rem; }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #2d2d44;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .price-box {
        background: linear-gradient(135deg, #e94560, #0f3460);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .section-header {
        color: #e94560;
        font-size: 1.3rem;
        font-weight: bold;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.3rem;
        margin: 1.5rem 0 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1.1rem;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ─── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    return obj["pipeline"], obj["metadata"]


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    if CLEAN_CSV.exists():
        return pd.read_csv(CLEAN_CSV)
    return preprocess_pipeline(data_dir=str(DATA_DIR))


# ─── Prediction helper ────────────────────────────────────────────────────────

def predict_price(pipeline, metadata: dict, inputs: dict) -> float:
    num_feats = metadata["numeric_features"]
    cat_feats = metadata["categorical_features"]
    all_feats = num_feats + cat_feats

    row = {f: [inputs.get(f, np.nan)] for f in all_feats}
    df_input = pd.DataFrame(row)

    # Apply same feature engineering
    df_input = engineer_features(df_input)

    # Make sure engineered cols are included
    for col in num_feats:
        if col not in df_input.columns:
            df_input[col] = np.nan
    for col in cat_feats:
        if col not in df_input.columns:
            df_input[col] = "Unknown"

    price = pipeline.predict(df_input[all_feats])[0]
    return max(0, price)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar_nav():
    st.sidebar.markdown("## 🚗 Navigation")
    page = st.sidebar.radio(
        "",
        ["🔮 Price Predictor", "📊 Data Explorer", "📈 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**CarDekho Price Predictor**\n\nML-powered used car valuation across 6 Indian cities.")
    return page


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_predictor(pipeline, metadata):
    st.markdown("""
    <div class="main-header">
        <h1>🚗 CarDekho Price Predictor</h1>
        <p>Instant ML-powered valuation for used cars across India</p>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("⚠️ Model not found. Run `python train.py` first to train and save the model.")
        return

    meta = metadata
    oem_list  = sorted(meta.get("oem_list", []))
    model_map = meta.get("model_map", {})
    uniq      = meta.get("unique_values", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">🏷️ Car Identity</div>', unsafe_allow_html=True)
        city = st.selectbox("City", sorted(uniq.get("city", ["Bangalore"])))
        oem  = st.selectbox("Brand (OEM)", oem_list)
        car_models = model_map.get(oem, [])
        model_sel = st.selectbox("Model", car_models if car_models else ["Unknown"])
        variant = st.text_input("Variant Name", "VXI")
        body_type = st.selectbox("Body Type", sorted(uniq.get("body_type", [])))

    with col2:
        st.markdown('<div class="section-header">⚙️ Specifications</div>', unsafe_allow_html=True)
        fuel_type    = st.selectbox("Fuel Type", sorted(uniq.get("fuel_type", [])))
        transmission = st.selectbox("Transmission", sorted(uniq.get("transmission", [])))
        model_year   = st.slider("Year of Manufacture", 2000, 2024, 2019)
        engine_cc    = st.number_input("Engine Displacement (CC)", 500, 5000, 1197, step=50)
        max_power    = st.number_input("Max Power (BHP)", 30.0, 600.0, 82.0, step=0.5)
        mileage      = st.number_input("Mileage (kmpl)", 5.0, 50.0, 18.0, step=0.5)

    with col3:
        st.markdown('<div class="section-header">📋 Additional Details</div>', unsafe_allow_html=True)
        kms_driven   = st.number_input("Kms Driven", 0, 500000, 45000, step=1000)
        owner_no     = st.selectbox("Number of Owners", [1, 2, 3, 4, 5])
        seats        = st.selectbox("Seats", [4, 5, 6, 7, 8, 9])
        insurance    = st.selectbox("Insurance", sorted(uniq.get("insurance_validity", ["Third Party Insurance"])))
        drive_type   = st.selectbox("Drive Type", sorted(uniq.get("drive_type", ["FWD"])))
        torque_nm    = st.number_input("Torque (Nm)", 50.0, 700.0, 115.0, step=5.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Price")

    if predict_btn:
        car_age = 2024 - model_year
        inputs = {
            "city":               city.lower(),
            "fuel_type":          fuel_type,
            "body_type":          body_type,
            "transmission":       transmission,
            "oem":                oem,
            "model":              model_sel,
            "model_year":         model_year,
            "car_age":            car_age,
            "kms_driven":         kms_driven,
            "engine_cc":          engine_cc,
            "max_power_bhp":      max_power,
            "mileage":            mileage,
            "owner_no":           owner_no,
            "seats":              seats,
            "torque_nm":          torque_nm,
            "insurance_validity": insurance,
            "drive_type":         drive_type,
            "length_mm":          np.nan,
            "width_mm":           np.nan,
            "height_mm":          np.nan,
        }
        with st.spinner("Calculating …"):
            price = predict_price(pipeline, metadata, inputs)

        price_lakh = price / 1e5
        low  = price_lakh * 0.92
        high = price_lakh * 1.08

        st.markdown(f"""
        <div class="price-box">
            Estimated Price: ₹ {price_lakh:.2f} Lakh
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Lower Bound (−8%)", f"₹ {low:.2f} L")
        c2.metric("Predicted Price",   f"₹ {price_lakh:.2f} L")
        c3.metric("Upper Bound (+8%)", f"₹ {high:.2f} L")

        st.info(
            f"**Summary:** A {model_year} {oem} {model_sel} ({transmission}, {fuel_type}) "
            f"with {kms_driven:,} km driven in {city.title()} is estimated at "
            f"**₹ {price_lakh:.2f} Lakh** (range ₹ {low:.2f}L – ₹ {high:.2f}L)."
        )


def page_explorer():
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    df = load_data()
    if df is None or df.empty:
        st.error("Dataset not available. Please run the preprocessing pipeline.")
        return

    st.markdown(f"**Total records:** {len(df):,}  |  **Columns:** {len(df.columns)}")

    tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Comparisons"])

    with tab1:
        st.subheader("Dataset Sample")
        show_cols = ["city", "oem", "model", "model_year", "fuel_type",
                     "transmission", "kms_driven", "price", "car_age"]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].head(50), use_container_width=True)

        st.subheader("Descriptive Statistics")
        st.dataframe(df.select_dtypes(include=np.number).describe().round(2),
                     use_container_width=True)

        st.subheader("City-wise Breakdown")
        city_stats = df.groupby("city").agg(
            Count=("price", "count"),
            Median_Price=("price", lambda x: round(x.median() / 1e5, 2)),
            Avg_Age=("car_age", "mean"),
        ).reset_index()
        city_stats.columns = ["City", "Listings", "Median Price (Lakh)", "Avg Age (yrs)"]
        st.dataframe(city_stats, use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["price"] / 1e5, bins=40, color="#e94560", edgecolor="white")
            ax.set(xlabel="Price (Lakh ₹)", ylabel="Count", title="Price Distribution")
            st.pyplot(fig)
            plt.close()

        with col_b:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df["kms_driven"] / 1000, bins=40, color="#0f3460", edgecolor="white")
            ax.set(xlabel="Kms Driven (K)", ylabel="Count", title="Kms Driven Distribution")
            st.pyplot(fig)
            plt.close()

        fig, ax = plt.subplots(figsize=(10, 4))
        df["fuel_type"].value_counts().plot(kind="bar", ax=ax, color="#16213e", edgecolor="#e94560")
        ax.set(title="Fuel Type Counts", xlabel="Fuel Type", ylabel="Count")
        plt.xticks(rotation=15)
        st.pyplot(fig)
        plt.close()

    with tab3:
        group_col = st.selectbox("Group by", ["city", "fuel_type", "transmission", "body_type"])
        metric    = st.selectbox("Metric", ["Median Price", "Mean Price", "Count"])

        if metric == "Median Price":
            grp = df.groupby(group_col)["price"].median() / 1e5
            ylabel = "Median Price (Lakh ₹)"
        elif metric == "Mean Price":
            grp = df.groupby(group_col)["price"].mean() / 1e5
            ylabel = "Mean Price (Lakh ₹)"
        else:
            grp = df.groupby(group_col)["price"].count()
            ylabel = "Count"

        grp = grp.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        grp.plot(kind="bar", ax=ax, color="#e94560", edgecolor="white")
        ax.set(title=f"{metric} by {group_col.replace('_',' ').title()}",
               xlabel=group_col.replace("_", " ").title(), ylabel=ylabel)
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close()


def page_model_insights(metadata):
    st.markdown('<div class="section-header">📈 Model Insights</div>', unsafe_allow_html=True)

    if metadata is None:
        st.error("Model metadata not available.")
        return

    # All models comparison
    all_results = metadata.get("all_results", {})
    if all_results:
        st.subheader("Model Comparison")
        rows = []
        for name, m in all_results.items():
            rows.append({
                "Model":         name,
                "R² Score":      round(m.get("R2", 0), 4),
                "MAE (Lakh ₹)":  round(m.get("MAE_lakh", 0), 2),
                "RMSE (Lakh ₹)": round(m.get("RMSE", 0) / 1e5, 2),
            })
        df_res = pd.DataFrame(rows).sort_values("R² Score", ascending=False)
        st.dataframe(df_res, use_container_width=True)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        df_res.plot(kind="bar", x="Model", y="R² Score",
                    ax=axes[0], color="#e94560", legend=False)
        axes[0].set(title="R² Score (higher better)", ylim=(0, 1))
        df_res.plot(kind="bar", x="Model", y="MAE (Lakh ₹)",
                    ax=axes[1], color="#0f3460", legend=False)
        axes[1].set(title="MAE – Lakh ₹ (lower better)")
        df_res.plot(kind="bar", x="Model", y="RMSE (Lakh ₹)",
                    ax=axes[2], color="#16213e", legend=False)
        axes[2].set(title="RMSE – Lakh ₹ (lower better)")
        for ax in axes:
            plt.sca(ax)
            plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Best model final metrics
    st.subheader("Best Model — Final Test Metrics")
    best  = metadata.get("best_model", "N/A")
    final = metadata.get("final_metrics", {})
    st.markdown(f"**Selected Model:** `{best}`")
    m1, m2, m3 = st.columns(3)
    m1.metric("R² Score",       f"{final.get('R2', 0):.4f}")
    m2.metric("MAE",            f"₹ {final.get('MAE_lakh', 0):.2f} Lakh")
    m3.metric("RMSE",           f"₹ {final.get('RMSE', 0)/1e5:.2f} Lakh")

    # Feature list
    st.subheader("Features Used")
    num_f = metadata.get("numeric_features", [])
    cat_f = metadata.get("categorical_features", [])
    fc1, fc2 = st.columns(2)
    fc1.markdown("**Numeric Features**\n" + "\n".join(f"- `{f}`" for f in num_f))
    fc2.markdown("**Categorical Features**\n" + "\n".join(f"- `{f}`" for f in cat_f))


def page_about():
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About</h1>
        <p>CarDekho Used Car Price Prediction Project</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ## Project Overview
    This project builds a **production-ready ML pipeline** to predict used car prices
    across 6 Indian cities using data scraped from CarDekho.

    ## Tech Stack
    | Component | Technology |
    |---|---|
    | Data Processing | Pandas, NumPy |
    | ML Framework | Scikit-Learn |
    | Models | Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting |
    | Visualisation | Matplotlib, Seaborn |
    | Web App | Streamlit |
    | Serialisation | Pickle |

    ## Cities Covered
    Bangalore · Chennai · Delhi · Hyderabad · Jaipur · Kolkata

    ## Evaluation Metrics
    - **MAE** – Mean Absolute Error
    - **MSE / RMSE** – Root Mean Squared Error
    - **R²** – Coefficient of Determination

    ## How to Run
    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Train the model
    python train.py

    # 3. Launch the app
    streamlit run app.py
    ```

    ## Project Structure
    ```
    cardekho/
    ├── app.py                  ← Streamlit application
    ├── train.py                ← Training entry point
    ├── requirements.txt
    ├── data/                   ← Raw & cleaned data
    ├── models/                 ← Serialised model
    ├── reports/figures/        ← EDA plots
    └── src/
        ├── data_preprocessing.py
        ├── eda.py
        └── model_training.py
    ```
    """)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    page = sidebar_nav()
    pipeline, metadata = load_model()

    if page == "🔮 Price Predictor":
        page_predictor(pipeline, metadata)
    elif page == "📊 Data Explorer":
        page_explorer()
    elif page == "📈 Model Insights":
        page_model_insights(metadata)
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
