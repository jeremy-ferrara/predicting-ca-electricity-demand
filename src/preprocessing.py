import pandas as pd
import numpy as np


TARGET_COL = "target_load_mw_mean"

LEAKAGE_RISK_COLS = [
    "load_mw_mean",
    "load_mw_max",
    "load_mw_min",
    "load_mw_std"
]

RAW_CALENDAR_COLS_TO_DROP = [
    "datetime",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_name",
    "day_of_year"
]


def load_data(filepath):
    df = pd.read_csv(filepath)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_lag_features(df):
    df = df.copy()

    # yesterday's load
    df["lag1_load"] = df[TARGET_COL].shift(1)

    return df


def add_aggregated_weather_features(df, base_temp=65, hot_temp=75, very_hot_temp=85):
    df = df.copy()

    temp_mean_cols = [col for col in df.columns if col.endswith("_temp_mean")]
    temp_max_cols = [col for col in df.columns if col.endswith("_temp_max")]
    temp_min_cols = [col for col in df.columns if col.endswith("_temp_min")]
    rhum_mean_cols = [col for col in df.columns if col.endswith("_rhum_mean")]
    prcp_sum_cols = [col for col in df.columns if col.endswith("_prcp_sum")]
    wspd_mean_cols = [col for col in df.columns if col.endswith("_wspd_mean")]

    if not temp_mean_cols:
        raise ValueError("No temperature mean columns found in input data.")

    df["temp_mean_all"] = df[temp_mean_cols].mean(axis=1)
    df["temp_max_all"] = df[temp_max_cols].mean(axis=1)
    df["temp_min_all"] = df[temp_min_cols].mean(axis=1)

    df["rhum_mean_all"] = df[rhum_mean_cols].mean(axis=1)
    df["prcp_sum_all"] = df[prcp_sum_cols].sum(axis=1)
    df["wspd_mean_all"] = df[wspd_mean_cols].mean(axis=1)

    df["temp_range_all"] = df["temp_max_all"] - df["temp_min_all"]

    df["cooling_degree"] = np.maximum(0, df["temp_mean_all"] - base_temp)
    df["heating_degree"] = np.maximum(0, base_temp - df["temp_mean_all"])

    df["cooling_degree_sq"] = df["cooling_degree"] ** 2
    df["heating_degree_sq"] = df["heating_degree"] ** 2

    df["is_hot"] = (df["temp_mean_all"] >= hot_temp).astype(int)
    df["is_very_hot"] = (df["temp_mean_all"] >= very_hot_temp).astype(int)

    return df


def add_calendar_features(df):
    df = df.copy()

    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_shoulder_season"] = df["month"].isin([3, 4, 5, 9, 10, 11]).astype(int)

    return df


def add_cyclical_features(df):
    df = df.copy()

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def add_interaction_features(df):
    df = df.copy()

    df["hot_weekend_interaction"] = df["cooling_degree"] * df["is_weekend"]
    df["cold_weekend_interaction"] = df["heating_degree"] * df["is_weekend"]

    df["summer_cooling_interaction"] = df["cooling_degree"] * df["is_summer"]
    df["winter_heating_interaction"] = df["heating_degree"] * df["is_winter"]

    df["humidity_heat_interaction"] = df["temp_mean_all"] * df["rhum_mean_all"]

    return df


def select_model_features(df):
    df = df.copy()

    feature_cols = [
        "lag1_load",
        "temp_mean_all",
        "temp_max_all",
        "temp_min_all",
        "temp_range_all",
        "rhum_mean_all",
        "prcp_sum_all",
        "wspd_mean_all",
        "cooling_degree",
        "heating_degree",
        "cooling_degree_sq",
        "heating_degree_sq",
        "is_hot",
        "is_very_hot",
        "is_weekend",
        "is_summer",
        "is_winter",
        "is_shoulder_season",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "hot_weekend_interaction",
        "cold_weekend_interaction",
        "summer_cooling_interaction",
        "winter_heating_interaction",
        "humidity_heat_interaction",
        TARGET_COL
    ]

    return df[feature_cols]


def preprocess_data(df):
    df_model = df.copy()

    df_model = add_lag_features(df_model)
    df_model = df_model.drop(columns=LEAKAGE_RISK_COLS, errors="ignore")
    df_model = add_aggregated_weather_features(df_model)
    df_model = add_calendar_features(df_model)
    df_model = add_cyclical_features(df_model)
    df_model = add_interaction_features(df_model)
    df_model = select_model_features(df_model)

    # drop first row created by lag1 shift
    df_model = df_model.dropna().reset_index(drop=True)

    return df_model


def split_data(df_model, target_col=TARGET_COL, train_frac=0.70, val_frac=0.15):
    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    n = len(df_model)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    X_train = X.iloc[:train_end].reset_index(drop=True)
    y_train = y.iloc[:train_end].reset_index(drop=True)

    X_val = X.iloc[train_end:val_end].reset_index(drop=True)
    y_val = y.iloc[train_end:val_end].reset_index(drop=True)

    X_test = X.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_model_data(filepath):
    df = load_data(filepath)
    df_model = preprocess_data(df)
    return split_data(df_model)