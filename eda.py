"""
CarDekho Used Car Price Prediction
Exploratory Data Analysis Module

Generates all EDA plots and saves them under the reports/figures/ directory.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PALETTE = "viridis"
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.1)
FIG_DIR = Path("reports/figures")


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# ─── Individual plots ─────────────────────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df["price"] / 1e5, bins=50, color="#4A90D9", edgecolor="white")
    axes[0].set(title="Price Distribution (Lakh ₹)", xlabel="Price (Lakh)", ylabel="Count")
    axes[1].hist(np.log1p(df["price"]), bins=50, color="#E87D3E", edgecolor="white")
    axes[1].set(title="Log-Price Distribution", xlabel="log(1 + Price)", ylabel="Count")
    fig.suptitle("Car Price Distributions", fontsize=14, fontweight="bold")
    _save(fig, "01_price_distribution")


def plot_price_by_city(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("city")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="city", y="price", order=order,
                palette="Set2", flierprops=dict(marker=".", markersize=2), ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.set(title="Price Distribution by City", xlabel="City", ylabel="Price")
    plt.xticks(rotation=20)
    _save(fig, "02_price_by_city")


def plot_price_by_fuel(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("fuel_type")["price"].median().sort_values(ascending=False).index
    sns.violinplot(data=df, x="fuel_type", y="price", order=order,
                   palette="muted", inner="quartile", ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.set(title="Price Distribution by Fuel Type", xlabel="Fuel Type", ylabel="Price")
    plt.xticks(rotation=15)
    _save(fig, "03_price_by_fuel")


def plot_price_by_transmission(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="transmission", y="price", palette="pastel", ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.set(title="Price by Transmission Type", xlabel="Transmission", ylabel="Price")
    _save(fig, "04_price_by_transmission")


def plot_kms_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(3000, len(df)), random_state=42)
    sc = ax.scatter(sample["kms_driven"] / 1000, sample["price"] / 1e5,
                    c=sample["car_age"], cmap="plasma", alpha=0.5, s=18)
    plt.colorbar(sc, ax=ax, label="Car Age (years)")
    ax.set(title="Kms Driven vs Price (coloured by Car Age)",
           xlabel="Kms Driven (thousands)", ylabel="Price (Lakh ₹)")
    _save(fig, "05_kms_vs_price")


def plot_car_age_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    avg = df.groupby("car_age")["price"].median().reset_index()
    ax.plot(avg["car_age"], avg["price"] / 1e5, marker="o", color="#2ecc71")
    ax.fill_between(avg["car_age"], avg["price"] / 1e5, alpha=0.15, color="#2ecc71")
    ax.set(title="Median Price vs Car Age", xlabel="Car Age (years)", ylabel="Median Price (Lakh ₹)")
    _save(fig, "06_car_age_vs_price")


def plot_top_brands(df: pd.DataFrame, top_n: int = 15):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    counts = df["oem"].value_counts().head(top_n)
    counts.plot(kind="barh", ax=axes[0], color="#3498db")
    axes[0].set(title=f"Top {top_n} Brands by Listing Count", xlabel="Count")
    axes[0].invert_yaxis()

    brand_price = df.groupby("oem")["price"].median().nlargest(top_n)
    brand_price.plot(kind="barh", ax=axes[1], color="#e74c3c")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    axes[1].set(title=f"Top {top_n} Brands by Median Price", xlabel="Median Price")
    axes[1].invert_yaxis()

    fig.suptitle("Brand Analysis", fontsize=14, fontweight="bold")
    _save(fig, "07_brand_analysis")


def plot_body_type(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("body_type")["price"].median().sort_values(ascending=False).index
    sns.barplot(data=df, x="body_type", y="price", order=order,
                estimator=np.median, palette="coolwarm", errorbar=None, ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.set(title="Median Price by Body Type", xlabel="Body Type", ylabel="Median Price (Lakh ₹)")
    plt.xticks(rotation=20)
    _save(fig, "08_body_type_price")


def plot_ownership_analysis(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = sorted(df["owner_no"].dropna().unique())
    df_plot = df[df["owner_no"].isin(order)]
    sns.boxplot(data=df_plot, x="owner_no", y="price", palette="Set3", ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.set(title="Price by Number of Owners", xlabel="Number of Owners", ylabel="Price (Lakh ₹)")
    _save(fig, "09_ownership_price")


def plot_correlation_heatmap(df: pd.DataFrame):
    num_df = df.select_dtypes(include=np.number).dropna(axis=1, how="all")
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    _save(fig, "10_correlation_heatmap")


def plot_missing_values(df: pd.DataFrame):
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        logger.info("No missing values to plot.")
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    miss.plot(kind="bar", ax=ax, color="#e67e22")
    ax.set(title="Missing Value Percentage by Column",
           xlabel="Column", ylabel="Missing %")
    plt.xticks(rotation=30)
    _save(fig, "00_missing_values")


def plot_mileage_vs_price(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=df.sample(min(2000, len(df)), random_state=1),
                    x="mileage", y="price", hue="fuel_type",
                    alpha=0.5, s=20, ax=ax)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e5:.0f}L"))
    ax.set(title="Mileage vs Price by Fuel Type",
           xlabel="Mileage (kmpl)", ylabel="Price (Lakh ₹)")
    _save(fig, "11_mileage_vs_price")


def generate_all_plots(df: pd.DataFrame):
    """Run all EDA plots in sequence."""
    logger.info("Generating all EDA plots …")
    plot_missing_values(df)
    plot_price_distribution(df)
    plot_price_by_city(df)
    plot_price_by_fuel(df)
    plot_price_by_transmission(df)
    plot_kms_vs_price(df)
    plot_car_age_vs_price(df)
    plot_top_brands(df)
    plot_body_type(df)
    plot_ownership_analysis(df)
    plot_correlation_heatmap(df)
    plot_mileage_vs_price(df)
    logger.info("All EDA plots saved.")


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe(include="all")


if __name__ == "__main__":
    from data_preprocessing import preprocess_pipeline
    df = preprocess_pipeline(data_dir="../data")
    generate_all_plots(df)
    print(descriptive_stats(df))
