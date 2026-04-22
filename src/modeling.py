import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_split_data(
    x_train_path,
    x_val_path,
    x_test_path,
    y_train_path,
    y_val_path,
    y_test_path
):
    X_train = pd.read_csv(x_train_path)
    X_val = pd.read_csv(x_val_path)
    X_test = pd.read_csv(x_test_path)

    y_train = pd.read_csv(y_train_path).squeeze("columns")
    y_val = pd.read_csv(y_val_path).squeeze("columns")
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_column_types(X_train):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }


def train_and_compare_models(X_train, y_train, X_val, y_val):
    numeric_cols, categorical_cols = get_column_types(X_train)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = get_models()

    results = []
    fitted_pipelines = {}

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        val_preds = pipeline.predict(X_val)

        metrics = evaluate_regression(y_val, val_preds)
        metrics["Model"] = model_name

        results.append(metrics)
        fitted_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return results_df, fitted_pipelines


def retrain_best_model(best_pipeline, X_train, y_train, X_val, y_val):
    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    best_pipeline.fit(X_train_full, y_train_full)

    return best_pipeline


def test_model(model, X_test, y_test):
    test_preds = model.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_preds)

    return test_preds, test_metrics


def summarize_results(best_model_name, val_metrics, test_metrics):
    summary_df = pd.DataFrame([{
        "Selected_Model": best_model_name,
        "Validation_MAE": val_metrics["MAE"],
        "Validation_RMSE": val_metrics["RMSE"],
        "Validation_R2": val_metrics["R2"],
        "Test_MAE": test_metrics["MAE"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_R2": test_metrics["R2"]
    }])

    return summary_df


def build_prediction_comparison(y_test, test_preds):
    pred_comparison = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": test_preds,
        "error": y_test.reset_index(drop=True) - test_preds,
        "abs_error": np.abs(y_test.reset_index(drop=True) - test_preds)
    })

    return pred_comparison


def save_model(model, filepath):
    joblib.dump(model, filepath)


def load_model(filepath):
    return joblib.load(filepath)