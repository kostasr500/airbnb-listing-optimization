import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_prepare_data(file_path: str):
    # load data and isolate features and target variable
    df = pd.read_csv(file_path)

    features = [
        "room_type",
        "neighbourhood_cleansed",
        "accommodates",
        "bedrooms",
        "beds",
        "minimum_nights",
        "number_of_reviews",
        "review_scores_rating",
        "is_superhost",
    ]
    target = "price"

    # only available columns are used for features
    available_cols = [col for col in features if col in df.columns]
    X = df[available_cols].copy()
    y = df[target].values

    return X, y


def build_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    # Construction of a preprocessor for categorical and numerical data
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray):
    # calculate predictions and evaluation metrics
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n================ {name} Evaluation ================")
    print(f"MAE  : €{mae:.2f} (Μέσο σφάλμα ανά διανυκτέρευση)")
    print(f"RMSE : €{rmse:.2f}")
    print(f"R²   : {r2:.4f} ({r2 * 100:.1f}% εξηγούμενη διακύμανση)")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def extract_rf_feature_importance(rf_pipeline: Pipeline, top_n: int = 10):
    # Extract the most important features from the Random Forest model
    rf_model = rf_pipeline.named_steps["regressor"]
    preprocessor = rf_pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = rf_model.feature_importances_

    # clean feature names
    clean_names = [
        name.replace("num__", "").replace("cat__neighbourhood_cleansed_", "Neighborhood: ").replace("cat__room_type_", "Type: ")
        for name in feature_names
    ]

    fi_df = pd.DataFrame({"Feature": clean_names, "Importance": importances})
    fi_df = fi_df.sort_values(by="Importance", ascending=False).head(top_n)

    print(f"\nTop {top_n} Value Drivers (Feature Importance):")
    print(fi_df.to_string(index=False))


def main():
    data_path = "data/processed/athens_listings_clean.csv"
    X, y = load_and_prepare_data(data_path)

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # Baseline Model: Ridge Regression
    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=10.0)),
        ]
    )
    ridge_pipeline.fit(X_train, y_train)
    evaluate_model("Ridge Regression (Baseline)", ridge_pipeline, X_test, y_test)

    # Advanced Model: Random Forest Regressor
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=16,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    evaluate_model("Random Forest (Advanced)", rf_pipeline, X_test, y_test)

    # Export Insights
    extract_rf_feature_importance(rf_pipeline, top_n=10)


if __name__ == "__main__":
    main()