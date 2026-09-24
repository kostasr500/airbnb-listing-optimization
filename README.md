# Athens Airbnb Listings: Analysis & Price Model

A practical data project analyzing 14,044 Airbnb listings in Athens, Greece, using public data from Inside Airbnb. Built to practice analytical SQL with DuckDB, machine learning pipelines with scikit-learn, and reporting in Google Looker Studio.

**Dashboard:** [Looker Studio Report](https://datastudio.google.com/reporting/45d8e8df-3628-4b66-85e1-33363a839eb0)

## Data

Data snapshot sourced from Inside Airbnb for Athens. Following data cleaning, missing values correction, and filtering extreme price values (>15 and <700), the working dataset contains 14,044 listings.

## Key Findings

- Entire homes/apartments represent 92.7% of all listings, with an average listed price of €130.25. Shared rooms represent about 0.2% of the market.
- Hotel rooms carry the highest average listed price (€268.69).
- The highest average listed prices are found in central neighborhoods: Rigillis, Zappeion, Acropolis, Kolonaki, and Commercial Triangle - Plaka.

## Price Model

Ridge regression (baseline) and Random Forest models were trained to predict nightly listing prices. Categorical features (`room_type`, `neighbourhood_cleansed`) were one-hot encoded, and numerical features median-imputed and scaled.

| Model | MAE | RMSE | R² |
| :--- | :---: | :---: | :---: |
| Ridge Regression (alpha=10.0) | €41.80 | €64.74 | 0.39 |
| Random Forest Regressor | €38.71 | €62.37 | 0.43 |

The highest feature importances in the Random Forest were `bedrooms`, `number_of_reviews`, `accommodates`, and `review_scores_rating`.

## SQL Queries

Analytical queries in `sql/` run directly on the cleaned dataset using DuckDB:
- `01_market_overview.sql`: Calculates average and median prices per neighborhood, filtered for areas with at least 30 listings (`HAVING COUNT(*) >= 30`).
- `02_host_performance.sql`: Compares prices between Superhosts and standard hosts across room types.
- `03_pricing_tiers.sql`: Ranks listings by price within each neighborhood using `DENSE_RANK()`.

## Project Structure

```text
airbnb-listing-optimization/
├── data/
│   ├── raw/                       # Raw Inside Airbnb data files
│   └── processed/                 # Cleaned dataset (14,044 listings)
├── notebooks/
│   └── 01_data_exploration.ipynb  # Cleaning, outlier handling, EDA
├── sql/
│   ├── 01_market_overview.sql
│   ├── 02_host_performance.sql
│   └── 03_pricing_tiers.sql
├── src/
│   └── train_pricing_model.py     # Training and evaluation script
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Clone repository
git clone https://github.com/kostasr500/airbnb-listing-optimization
cd airbnb-listing-optimization

# Environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run pricing models
python3 src/train_pricing_model.py

# Run an SQL analysis file via DuckDB
python3 -c "import duckdb; duckdb.sql(open('sql/01_market_overview.sql').read()).show()"
```

## Tools

Python (Pandas, NumPy, scikit-learn), DuckDB, Jupyter, Google Looker Studio.
