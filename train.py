"""
CarDekho Used Car Price Prediction
Training Entry Point — train.py

Usage:
    python train.py
    python train.py --data-dir /path/to/data --model-path models/best_model.pkl --eda
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser(description="Train CarDekho price prediction model")
    p.add_argument("--data-dir",   default=str(ROOT / "data"),   help="Directory with city xlsx files")
    p.add_argument("--model-path", default=str(ROOT / "models" / "best_model.pkl"), help="Output model path")
    p.add_argument("--eda",        action="store_true", help="Generate EDA plots")
    p.add_argument("--eda-dir",    default=str(ROOT / "reports" / "figures"), help="EDA output directory")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  CarDekho Used Car Price Prediction – Training Pipeline")
    logger.info("=" * 60)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    logger.info("\n[1/3] Data Preprocessing …")
    from data_preprocessing import preprocess_pipeline
    clean_csv = str(Path(args.data_dir) / "cleaned_cars.csv")
    df = preprocess_pipeline(data_dir=args.data_dir, save_path=clean_csv)
    logger.info(f"      Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Step 2: EDA (optional) ───────────────────────────────────────────────
    if args.eda:
        logger.info("\n[2/3] Generating EDA plots …")
        import eda as eda_module
        eda_module.FIG_DIR = Path(args.eda_dir)
        eda_module.generate_all_plots(df)
        logger.info(f"      Plots saved to {args.eda_dir}")
    else:
        logger.info("\n[2/3] Skipping EDA (pass --eda to enable)")

    # ── Step 3: Model training ───────────────────────────────────────────────
    logger.info("\n[3/3] Model Training & Evaluation …")
    from model_training import run_training
    pipeline, metadata = run_training(data_dir=args.data_dir, model_path=args.model_path)

    logger.info("\n" + "=" * 60)
    logger.info("  Training complete! 🎉")
    logger.info(f"  Best model : {metadata['best_model']}")
    fm = metadata["final_metrics"]
    logger.info(f"  R²         : {fm['R2']:.4f}")
    logger.info(f"  MAE        : ₹ {fm['MAE_lakh']:.2f} Lakh")
    logger.info(f"  RMSE       : ₹ {fm['RMSE']/1e5:.2f} Lakh")
    logger.info(f"\n  Run the app:  streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
