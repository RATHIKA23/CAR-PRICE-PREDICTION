"""
CarDekho Used Car Price Prediction
Data Preprocessing Module

Converts raw nested JSON data from city-wise Excel files into a clean,
structured DataFrame ready for EDA and ML modelling.
"""

import ast
import re
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_parse(value):
    """Safely parse a stringified Python dict / list."""
    if pd.isna(value):
        return {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        return {}


def _extract_number(text):
    """Extract first numeric value from a string like '₹ 4 Lakh' or '23.1 kmpl'."""
    if pd.isna(text) or text == "":
        return np.nan
    nums = re.findall(r"[\d,]+\.?\d*", str(text).replace(",", ""))
    return float(nums[0]) if nums else np.nan


def _parse_price(price_str):
    """Convert '₹ 4 Lakh' → 400000, '₹ 12.50 Lakh' → 1250000."""
    if pd.isna(price_str) or price_str == "":
        return np.nan
    price_str = str(price_str).replace("₹", "").replace(",", "").strip()
    if "lakh" in price_str.lower():
        num = re.findall(r"[\d.]+", price_str)
        return float(num[0]) * 1e5 if num else np.nan
    if "crore" in price_str.lower():
        num = re.findall(r"[\d.]+", price_str)
        return float(num[0]) * 1e7 if num else np.nan
    num = re.findall(r"[\d.]+", price_str)
    return float(num[0]) if num else np.nan


def _kv_list_to_dict(lst):
    """[{'key': 'A', 'value': 'B'}, …] → {'A': 'B', …}"""
    result = {}
    for item in lst if isinstance(lst, list) else []:
        k = item.get("key") or item.get("heading")
        v = item.get("value")
        if k:
            result[str(k).strip()] = str(v).strip() if v is not None else np.nan
    return result


# ─── Per-row parsers ──────────────────────────────────────────────────────────

def parse_detail(raw):
    d = _safe_parse(raw)
    km = str(d.get("km", "")).replace(",", "")
    km_val = _extract_number(km)
    return {
        "fuel_type":    d.get("ft"),
        "body_type":    d.get("bt"),
        "kms_driven":   km_val,
        "transmission": d.get("transmission"),
        "owner_no":     d.get("ownerNo"),
        "owner":        d.get("owner"),
        "oem":          d.get("oem"),
        "model":        d.get("model"),
        "model_year":   d.get("modelYear"),
        "variant_name": d.get("variantName"),
        "price":        _parse_price(d.get("price")),
    }


def parse_overview(raw):
    d = _safe_parse(raw)
    top = _kv_list_to_dict(d.get("top", []))
    reg_year = top.get("Registration Year") or top.get("Year of Manufacture")
    seats = top.get("Seats", "")
    seats_val = _extract_number(seats)
    color = None
    for section in d.get("data", []):
        sub_dict = _kv_list_to_dict(section.get("list", []))
        if "Color" in sub_dict:
            color = sub_dict["Color"]
    return {
        "registration_year": _extract_number(reg_year),
        "insurance_validity": top.get("Insurance Validity"),
        "seats":             seats_val,
        "rto":               top.get("RTO"),
        "color":             color,
    }


def parse_specs(raw):
    d = _safe_parse(raw)
    top = _kv_list_to_dict(d.get("top", []))
    # Also flatten all sections
    all_kv = {}
    for section in d.get("data", []):
        all_kv.update(_kv_list_to_dict(section.get("list", [])))

    mileage = top.get("Mileage") or all_kv.get("Mileage")
    engine  = top.get("Engine") or all_kv.get("Displacement")
    power   = top.get("Max Power") or all_kv.get("Max Power")
    torque  = top.get("Torque") or all_kv.get("Max Torque")
    seats   = top.get("Seats") or all_kv.get("Seating Capacity")
    length  = all_kv.get("Length")
    width   = all_kv.get("Width")
    height  = all_kv.get("Height")
    gear_box = all_kv.get("Gear Box")
    drive_type = all_kv.get("Drive Type")

    return {
        "mileage":     _extract_number(mileage),
        "engine_cc":   _extract_number(engine),
        "max_power_bhp": _extract_number(power),
        "torque_nm":   _extract_number(torque),
        "length_mm":   _extract_number(length),
        "width_mm":    _extract_number(width),
        "height_mm":   _extract_number(height),
        "gear_box":    str(gear_box).strip() if gear_box else np.nan,
        "drive_type":  drive_type,
    }


# ─── City loader ──────────────────────────────────────────────────────────────

def load_city(filepath: str, city_name: str) -> pd.DataFrame:
    logger.info(f"Loading {city_name} from {filepath}")
    df_raw = pd.read_excel(filepath)

    records = []
    for _, row in df_raw.iterrows():
        rec = {"city": city_name, "car_link": row.get("car_links")}
        rec.update(parse_detail(row.get("new_car_detail")))
        rec.update(parse_overview(row.get("new_car_overview")))
        rec.update(parse_specs(row.get("new_car_specs")))
        records.append(rec)

    return pd.DataFrame(records)


# ─── Full pipeline ─────────────────────────────────────────────────────────────

CITIES = {
    "bangalore":  "bangalore_cars.xlsx",
    "chennai":    "chennai_cars.xlsx",
    "delhi":      "delhi_cars.xlsx",
    "hyderabad":  "hyderabad_cars.xlsx",
    "jaipur":     "jaipur_cars.xlsx",
    "kolkata":    "kolkata_cars.xlsx",
}


def load_all_cities(data_dir: str = "data") -> pd.DataFrame:
    """Load and concatenate all city datasets."""
    data_dir = Path(data_dir)
    frames = []
    for city, filename in CITIES.items():
        fp = data_dir / filename
        if fp.exists():
            frames.append(load_city(str(fp), city))
        else:
            logger.warning(f"File not found: {fp}")
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline."""
    logger.info("Starting data cleaning …")
    df = df.copy()

    # ── Drop rows with no price (target variable) ─────────────────────────────
    initial = len(df)
    df = df[df["price"].notna() & (df["price"] > 0)]
    logger.info(f"Dropped {initial - len(df)} rows with missing/zero price")

    # ── model_year: fill with registration_year if missing ───────────────────
    df["model_year"] = df["model_year"].fillna(df["registration_year"])
    df["car_age"] = 2024 - df["model_year"]
    df["car_age"] = df["car_age"].clip(lower=0)

    # ── Numeric columns: median imputation ───────────────────────────────────
    num_cols = ["kms_driven", "mileage", "engine_cc", "max_power_bhp",
                "torque_nm", "length_mm", "width_mm", "height_mm",
                "seats", "owner_no", "car_age"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # ── Categorical columns: mode imputation ─────────────────────────────────
    cat_cols = ["fuel_type", "body_type", "transmission", "owner",
                "oem", "model", "city", "insurance_validity",
                "color", "drive_type"]
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "Unknown")

    # ── Standardise strings ──────────────────────────────────────────────────
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # ── Outlier removal via IQR on price ────────────────────────────────────
    Q1 = df["price"].quantile(0.05)
    Q3 = df["price"].quantile(0.95)
    IQR = Q3 - Q1
    lower = max(0, Q1 - 1.5 * IQR)
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df["price"] >= lower) & (df["price"] <= upper)]
    logger.info(f"Removed {before - len(df)} price outliers (IQR method)")

    # ── Outlier cap on kms_driven ────────────────────────────────────────────
    km_upper = df["kms_driven"].quantile(0.99)
    df["kms_driven"] = df["kms_driven"].clip(upper=km_upper)

    # ── Drop duplicates ──────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["oem", "model", "model_year",
                                     "kms_driven", "fuel_type",
                                     "transmission", "city"])
    logger.info(f"Removed {before - len(df)} duplicate rows")

    logger.info(f"Clean dataset shape: {df.shape}")
    return df.reset_index(drop=True)


def preprocess_pipeline(data_dir: str = "data",
                         save_path: str = None) -> pd.DataFrame:
    """End-to-end: load → clean → (optionally save)."""
    df = load_all_cities(data_dir)
    df = clean_data(df)
    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved cleaned data to {save_path}")
    return df


if __name__ == "__main__":
    df = preprocess_pipeline(data_dir="data", save_path="data/cleaned_cars.csv")
    print(df.head())
    print(df.dtypes)
    print(df.describe())
