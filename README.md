# CAR-PRICE-PREDICTION
To create an accurate and user-friendly streamlit tool that predicts the prices of used cars based on various features. This tool should be deployed as an interactive web application for both customers and sales representatives to use seamlessly.
🚗 CarDekho — Used Car Price Prediction
> A production-ready ML system that predicts used car prices across 6 Indian cities, deployed as a Streamlit web application.
---
📋 Project Summary
Field	Details
Domain	Automotive Industry, Data Science, Machine Learning
Dataset	~8,400 used car listings (Bangalore, Chennai, Delhi, Hyderabad, Jaipur, Kolkata)
Target	Used car selling price (₹)
Best Model	Random Forest Regressor
R² Score	~0.93
MAE	~₹1.15 Lakh
---
🗂️ Project Structure
```
cardekho/
├── app.py                      ← Streamlit web application (4 pages)
├── train.py                    ← End-to-end training entry point
├── requirements.txt
├── README.md
│
├── data/
│   ├── bangalore\_cars.xlsx     ← Raw city data (6 cities)
│   ├── chennai\_cars.xlsx
│   ├── delhi\_cars.xlsx
│   ├── hyderabad\_cars.xlsx
│   ├── jaipur\_cars.xlsx
│   ├── kolkata\_cars.xlsx
│   └── cleaned\_cars.csv        ← Auto-generated after training
│
├── models/
│   ├── best\_model.pkl          ← Serialised sklearn Pipeline
│   └── best\_model.json         ← Model metadata (metrics, feature lists, unique values)
│
├── reports/
│   └── figures/                ← 12 EDA plots (auto-generated)
│
├── src/
│   ├── data\_preprocessing.py   ← Parse nested JSON → structured DataFrame + cleaning
│   ├── eda.py                  ← 12 EDA visualisations
│   └── model\_training.py       ← Feature engineering, Pipeline, training, evaluation
│
└── tests/
    └── test\_pipeline.py        ← 13 unit tests (pytest)
```
---
⚙️ Setup & Installation
1. Prerequisites
Python 3.9+
pip
2. Clone / download the project
```bash
git clone <your-repo-url>
cd cardekho
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Train the model
```bash
# Basic training
python train.py

# Training + EDA plots
python train.py --eda

# Custom paths
python train.py --data-dir /path/to/data --model-path models/my\_model.pkl --eda
```
5. Launch the Streamlit app
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
---
🔄 Pipeline Overview
Data Preprocessing (`src/data\_preprocessing.py`)
Each city Excel file contains 5 columns of nested Python-dict strings:
`new\_car\_detail` — price, fuel type, km, OEM, model, year, transmission
`new\_car\_overview` — registration year, insurance, seats, color, RTO
`new\_car\_specs` — mileage, engine CC, max power, torque, dimensions
`new\_car\_feature` — feature lists (not used for ML)
`car\_links` — URL
Steps:
Parse all JSON strings with `ast.literal\_eval`
Extract structured fields into flat columns
Add `city` column per dataset
Concatenate all 6 cities
Handle missing values (median for numeric, mode for categorical)
Standardise data formats (strip units, convert types)
Remove price outliers (IQR method, 5th–95th percentile)
Remove duplicate listings
Create `car\_age` feature
Feature Engineering (`src/model\_training.py`)
Additional derived features:
`power\_per\_cc` — BHP per CC (power density)
`kms\_per\_year` — kms driven normalised by age
`volume\_cm3` — length × width × height
`log\_kms\_driven`, `log\_engine\_cc` — log transforms of skewed features
Model Training
Model	R²	MAE (Lakh)
Ridge Regression	~0.72	~2.5
Lasso Regression	~0.70	~2.6
Decision Tree	~0.82	~1.8
Random Forest	~0.93	~1.15
Gradient Boosting	~0.91	~1.3
Selected: Random Forest — best R² with good generalisability.
Preprocessing Pipeline:
Numeric: `StandardScaler`
Categorical: `OneHotEncoder(handle\_unknown='ignore')`
Hyperparameter tuning: `RandomizedSearchCV` with 5-fold CV (on best model).
Evaluation Metrics
MAE — Mean Absolute Error (interpretable in ₹)
MSE / RMSE — Root Mean Squared Error
R² — Coefficient of Determination (variance explained)
---
🖥️ Streamlit App Pages
Page	Description
🔮 Price Predictor	Input car details → instant price estimate with confidence range
📊 Data Explorer	Browse dataset, distributions, city comparisons
📈 Model Insights	Compare all models, view final metrics and feature lists
ℹ️ About	Project documentation
---
🧪 Running Tests
```bash
pytest tests/ -v
```
13 unit tests covering parsers, data cleaning, and feature engineering.
---
🚀 Deployment Options
Local
```bash
streamlit run app.py
```
Streamlit Cloud
Push to GitHub
Connect repo at share.streamlit.io
Set `app.py` as entry point
Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python train.py
EXPOSE 8501
CMD \["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
---
📊 EDA Plots Generated
#	Plot
00	Missing values heatmap
01	Price distribution (raw + log)
02	Price by city (box plot)
03	Price by fuel type (violin)
04	Price by transmission
05	Kms driven vs price (scatter, coloured by age)
06	Median price vs car age
07	Top brands by count and median price
08	Body type vs price
09	Price by number of owners
10	Correlation heatmap
11	Mileage vs price by fuel type
---
🛠️ Technical Stack
Layer	Technology
Language	Python 3.9+
Data	Pandas, NumPy
ML	Scikit-Learn
Visualisation	Matplotlib, Seaborn
Web App	Streamlit
Serialisation	Pickle
Testing	Pytest
Code Style	PEP 8
---
📝 Methodology Notes
Why Random Forest?
Handles mixed data types natively
Robust to outliers without extensive preprocessing
Captures non-linear relationships (price vs age, km)
Built-in feature importance
Generalises well without aggressive regularisation
Regularisation: Ridge/Lasso included for linear baseline; RF implicitly regularises through bagging and feature subsampling.
Feature Selection: All parsed fields included + engineered features; OHE on categoricals; low-variance columns drop naturally via tree splits.
---
Built for CarDekho – Used Car Price Prediction Capstone Project
