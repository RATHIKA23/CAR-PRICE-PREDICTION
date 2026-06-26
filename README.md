# CAR-PRICE-PREDICTION

🚗 **CarDekho — Used Car Price Prediction**

> A production-ready ML system that predicts used car prices across 6 Indian cities, deployed as a Streamlit web application.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red)](http://localhost:8501)

---

## ⚡ Quick Start

Get up and running in 3 steps:

```bash
# 1. Clone the repository
git clone https://github.com/RATHIKA23/CAR-PRICE-PREDICTION.git
cd CAR-PRICE-PREDICTION

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model and launch app
python train.py
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Done! 🎉

---

## 📋 Project Summary

| Field | Details |
|-------|---------|
| **Domain** | Automotive Industry, Data Science, Machine Learning |
| **Dataset** | ~8,400 used car listings across 6 Indian cities |
| **Cities** | Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata |
| **Target Variable** | Used car selling price (₹) |
| **Best Model** | Random Forest Regressor |
| **R² Score** | ~0.93 (93% variance explained) |
| **MAE** | ~₹1.15 Lakh |
| **RMSE** | ~₹1.8 Lakh |
| **Model Size** | ~45 MB (pickled) |

---

## ✨ Key Features

- ✅ **Multi-city Support**: Trained on 6 major Indian cities with regional insights
- ✅ **Production-Ready**: Fully tested pipeline with 13 unit tests
- ✅ **Advanced Feature Engineering**: 8+ engineered features capturing domain knowledge
- ✅ **Interactive Web App**: 4-page Streamlit interface for predictions & exploration
- ✅ **Comprehensive EDA**: 12 automated visualization plots
- ✅ **Model Comparison**: Side-by-side evaluation of 5 ML algorithms
- ✅ **Confidence Intervals**: Prediction ranges for better decision-making
- ✅ **Easy Deployment**: Docker, Streamlit Cloud, or local support

---

## 🗂️ Project Structure

```
cardekho/
├── app.py                      ← Streamlit web application (4 pages)
├── train.py                    ← End-to-end training entry point
├── requirements.txt            ← Python dependencies
├── README.md
│
├── data/
│   ├── bangalore_cars.xlsx     ← Raw city data (6 cities)
│   ├── chennai_cars.xlsx
│   ├── delhi_cars.xlsx
│   ├── hyderabad_cars.xlsx
│   ├── jaipur_cars.xlsx
│   ├── kolkata_cars.xlsx
│   └── cleaned_cars.csv        ← Auto-generated after training
│
├── models/
│   ├── best_model.pkl          ← Serialised sklearn Pipeline
│   └── best_model.json         ← Model metadata (metrics, feature lists, unique values)
│
├── reports/
│   └── figures/                ← 12 EDA plots (auto-generated)
│       ├── 00_missing_values.png
│       ├── 01_price_distribution.png
│       ├── ... (10 more plots)
│
├── src/
│   ├── data_preprocessing.py   ← Parse nested JSON → structured DataFrame + cleaning
│   ├── eda.py                  ← 12 EDA visualisations
│   └── model_training.py       ← Feature engineering, Pipeline, training, evaluation
│
└── tests/
    └── test_pipeline.py        ← 13 unit tests (pytest)
```

---

## 🖥️ Streamlit App Pages

| Page | Description | Use Case |
|------|-------------|----------|
| 🔮 **Price Predictor** | Input car details → instant price estimate with 95% confidence range | Customers, dealers |
| 📊 **Data Explorer** | Browse dataset, distributions, city comparisons, brand rankings | Market analysis, research |
| 📈 **Model Insights** | Compare all 5 models, view final metrics, feature importance, training logs | Model evaluation, debugging |
| ℹ️ **About** | Project documentation, methodology, technical stack | Reference |

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- ~2 GB disk space
- Virtual environment (recommended)

### 2. Clone the Repository

```bash
git clone https://github.com/RATHIKA23/CAR-PRICE-PREDICTION.git
cd CAR-PRICE-PREDICTION
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the Model

```bash
# Basic training (quick)
python train.py

# Training + generate EDA plots
python train.py --eda

# Custom paths
python train.py --data-dir ./data --model-path ./models/my_model.pkl --eda

# View all options
python train.py --help
```

**Training Time**: ~2–3 minutes (depends on CPU)

### 6. Launch the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🔄 Pipeline Overview

### Data Preprocessing (`src/data_preprocessing.py`)

Each city Excel file contains 5 columns of nested Python-dict strings:

- `new_car_detail` — price, fuel type, km, OEM, model, year, transmission
- `new_car_overview` — registration year, insurance, seats, color, RTO
- `new_car_specs` — mileage, engine CC, max power, torque, dimensions
- `new_car_feature` — feature lists (not used for ML)
- `car_links` — URL

**Processing Steps:**
1. Parse all JSON strings with `ast.literal_eval`
2. Extract structured fields into flat columns
3. Add `city` column per dataset
4. Concatenate all 6 cities
5. Handle missing values (median for numeric, mode for categorical)
6. Standardise data formats (strip units, convert types)
7. Remove price outliers (IQR method, 5th–95th percentile)
8. Remove duplicate listings
9. Create `car_age` feature from registration year

**Output**: `data/cleaned_cars.csv` (~8,000 rows, 20+ features)

### Feature Engineering (`src/model_training.py`)

**Derived Features:**
- `power_per_cc` — BHP per CC (power density indicator)
- `kms_per_year` — kms driven normalised by age
- `volume_cm3` — cargo volume (length × width × height)
- `log_kms_driven`, `log_engine_cc` — log transforms of skewed features

**Rationale**: Captures domain knowledge about car market (newer cars, powerful cars, fuel-efficient variants command premium prices)

### Model Training & Evaluation

**Model Comparison**:

| Model | R² | MAE (Lakh) | RMSE (Lakh) | Training Time |
|-------|-----|-----------|------------|--------------|
| Ridge Regression | 0.72 | 2.50 | 3.2 | <1s |
| Lasso Regression | 0.70 | 2.60 | 3.4 | <1s |
| Decision Tree | 0.82 | 1.80 | 2.5 | 2s |
| Gradient Boosting | 0.91 | 1.30 | 1.9 | 45s |
| **Random Forest** | **0.93** | **1.15** | **1.6** | **30s** |

**Selected**: Random Forest — best R² with good generalisability and feature importance interpretability.

**Pipeline Architecture**:
```
StandardScaler (numeric) + OneHotEncoder (categorical)
    ↓
Random Forest Regressor (n_estimators=100, max_depth=20)
    ↓
Hyperparameter Tuning (RandomizedSearchCV, 5-fold CV)
```

**Evaluation Metrics**:
- **MAE**: Mean Absolute Error (interpretable in rupees)
- **RMSE**: Root Mean Squared Error (penalizes large errors)
- **R²**: Coefficient of Determination (% variance explained)
- **CV Score**: 5-fold cross-validation for stability

---

## 📊 EDA Plots Generated

All plots are saved in `reports/figures/` after running `python train.py --eda`:

| # | Plot | Insight |
|---|------|---------|
| 00 | Missing values heatmap | Data quality & gaps |
| 01 | Price distribution (raw + log) | Target variable skewness |
| 02 | Price by city (box plot) | Regional price variations |
| 03 | Price by fuel type (violin) | Fuel efficiency premium |
| 04 | Price by transmission | Manual vs automatic preferences |
| 05 | Kms driven vs price (scatter, coloured by age) | Depreciation curve |
| 06 | Median price vs car age | Age-price relationship |
| 07 | Top brands by count and median price | Market dominance & brand value |
| 08 | Body type vs price | Segment preferences |
| 09 | Price by number of owners | Ownership history impact |
| 10 | Correlation heatmap | Feature relationships |
| 11 | Mileage vs price by fuel type | Fuel type interaction effects |

---

## 🧪 Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_pipeline.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

**Test Coverage**: 13 unit tests covering:
- JSON parsing & data extraction
- Data cleaning & outlier removal
- Feature engineering
- Pipeline end-to-end

All tests should pass in <5 seconds.

---

## 🚀 Deployment Options

### Option 1: Local (Development)

```bash
streamlit run app.py
```

Best for: Development, testing, small teams

### Option 2: Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `app.py` as entry point
4. Deploy in 2 clicks

Best for: Quick public sharing, no server costs

### Option 3: Docker (Production)

Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN python train.py

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t cardekho:latest .
docker run -p 8501:8501 cardekho:latest
```

Best for: Scalability, reproducibility, cloud deployment (AWS, GCP, Azure)

### Option 4: Other Cloud Platforms

- **Heroku**: Add `Procfile` with `web: streamlit run app.py`
- **AWS EC2**: Deploy Docker container
- **Google Cloud Run**: Containerized deployment with auto-scaling

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Visualisation** | Matplotlib, Seaborn, Plotly |
| **Web App** | Streamlit |
| **Serialisation** | Pickle, JSON |
| **Testing** | Pytest, Coverage |
| **Code Quality** | PEP 8, Type hints |

**Requirements Summary**:
- `pandas>=1.3.0` — Data manipulation
- `scikit-learn>=1.0.0` — ML algorithms
- `streamlit>=1.20.0` — Web app framework
- `matplotlib>=3.5.0` — Static plots
- `seaborn>=0.12.0` — Statistical plots
- `numpy>=1.21.0` — Numerical computing

---

## 📝 Methodology Notes

### Why Random Forest?

✅ **Handles mixed data types** natively (no separate encoding pipeline)  
✅ **Robust to outliers** without extensive preprocessing  
✅ **Captures non-linear relationships** (price vs age, km, transmission)  
✅ **Built-in feature importance** for interpretability  
✅ **Generalises well** without aggressive regularisation  
✅ **Fast predictions** (~5ms per car)  

### Feature Selection Strategy

- All parsed fields included + 5 engineered features
- One-Hot Encoding on categorical features
- Low-variance columns dropped naturally via tree splits
- No explicit feature selection needed (tree-based pruning)

### Regularisation Approach

- Ridge/Lasso included as baselines for linear comparison
- Random Forest implicitly regularises through:
  - Bagging (bootstrap samples)
  - Feature subsampling (√features per split)
  - Max depth constraints

### Cross-Validation

- 5-fold stratified CV on training set
- Ensures model stability across city subsets
- Reports mean ± std of metrics

---

## 🆘 Troubleshooting

### Issue: "Port 8501 already in use"
```bash
streamlit run app.py --server.port=8502
```

### Issue: "Data files not found"
Ensure all 6 Excel files are in `data/` directory before running `train.py`:
- `bangalore_cars.xlsx`
- `chennai_cars.xlsx`
- `delhi_cars.xlsx`
- `hyderabad_cars.xlsx`
- `jaipur_cars.xlsx`
- `kolkata_cars.xlsx`

### Issue: "Model not found" when running app
```bash
python train.py  # Regenerate models
```

### Issue: Slow predictions on large datasets
- Use `--data-dir` to limit dataset size during training
- Reduce `n_estimators` in `model_training.py` (line ~80)

### Issue: Memory error during training
- Reduce batch size in data preprocessing
- Train on subset of cities first

For more help, open an [issue on GitHub](https://github.com/RATHIKA23/CAR-PRICE-PREDICTION/issues).

---

## 💡 Future Enhancements

- [ ] Add XGBoost/LightGBM models for speed benchmarking
- [ ] Implement hyperparameter tuning with Optuna
- [ ] Add SHAP values for prediction explanations
- [ ] Deploy to AWS SageMaker for scalable inference
- [ ] Collect real-time pricing data via web scraping
- [ ] Add multi-language support to Streamlit app
- [ ] Implement A/B testing for model updates

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this project for personal and commercial purposes.

---

## 👤 Author

**Rathika** — [@RATHIKA23](https://github.com/RATHIKA23)

- 📧 Email: [your-email@example.com](mailto:your-email@example.com)
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)
- 🔗 GitHub: [RATHIKA23](https://github.com/RATHIKA23)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guide
- Tests pass (`pytest tests/ -v`)
- New features include unit tests
- README is updated with new changes

---

## 📞 Support & Feedback

Have questions or suggestions? 

- 🐛 [Report a Bug](https://github.com/RATHIKA23/CAR-PRICE-PREDICTION/issues/new?template=bug_report.md)
- ✨ [Request a Feature](https://github.com/RATHIKA23/CAR-PRICE-PREDICTION/issues/new?template=feature_request.md)
- 💬 [Start a Discussion](https://github.com/RATHIKA23/CAR-PRICE-PREDICTION/discussions)

---

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn User Guide](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [CarDekho Dataset](https://www.kaggle.com/datasets/nehalbirpau/cardekho-used-cars-dataset)

---

**Built with ❤️ for the CarDekho — Used Car Price Prediction Capstone Project**

Last updated: 2026-06-26 | Version: 1.1.0
