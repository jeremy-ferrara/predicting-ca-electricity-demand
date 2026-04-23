import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import numpy as np
import requests
import streamlit as st

try:
    import gridstatus
    GRIDSTATUS_IMPORT_ERROR = None
except Exception as e:
    gridstatus = None
    GRIDSTATUS_IMPORT_ERROR = e

st.set_page_config(page_title="CA Electricity Demand Forecast", page_icon="⚡", layout="wide")

MODEL_PATH = "ca_electricity_demand_lr_v1.joblib"
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY", ""))

CITY_COORDS = {
    "la": {"lat": 34.0522, "lon": -118.2437},
    "sf": {"lat": 37.7749, "lon": -122.4194},
    "sd": {"lat": 32.7157, "lon": -117.1611},
    "sj": {"lat": 37.3382, "lon": -121.8863},
    "fresno": {"lat": 36.7378, "lon": -119.7871},
}

BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
DISPLAY_CITY_FOR_ICON = "la"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def get_expected_feature_columns(model):
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.transformers_[0][2]


@st.cache_data(ttl=60 * 30)
def get_forecast_json(lat, lon, api_key, units="metric"):
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": units,
    }
    response = requests.get(BASE_URL, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def forecast_json_to_df(forecast_json):
    rows = []
    for item in forecast_json["list"]:
        rows.append({
            "datetime": pd.to_datetime(item["dt_txt"]),
            "temp": item["main"].get("temp"),
            "rhum": item["main"].get("humidity"),
            "wspd": item["wind"].get("speed"),
            "prcp": item.get("rain", {}).get("3h", 0.0) + item.get("snow", {}).get("3h", 0.0),
            "weather_main": item["weather"][0].get("main", ""),
            "weather_desc": item["weather"][0].get("description", ""),
            "weather_icon": item["weather"][0].get("icon", ""),
        })
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def prep_forecast_weather(df, prefix):
    keep = [
        "datetime",
        "temp",
        "rhum",
        "prcp",
        "wspd",
        "weather_main",
        "weather_desc",
        "weather_icon",
    ]
    df = df[keep].copy()
    return df.rename(columns={c: f"{prefix}_{c}" for c in df.columns if c != "datetime"})


def merge_city_forecasts(city_weather_dfs):
    merged = None
    for city_name, df in city_weather_dfs.items():
        city_df = prep_forecast_weather(df, city_name)
        if merged is None:
            merged = city_df
        else:
            merged = merged.merge(city_df, on="datetime", how="outer")
    return merged.sort_values("datetime").reset_index(drop=True)


def filter_full_forecast_days(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    df = df[df["datetime"] >= tomorrow].copy()
    df["date"] = df["datetime"].dt.normalize()

    counts = df.groupby("date").size()
    full_dates = counts[counts == 8].index

    df = df[df["date"].isin(full_dates)].copy()
    return df.drop(columns="date")


def build_daily_forecast_features(model_like_df):
    df = model_like_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = filter_full_forecast_days(df)

    weather_cols = [c for c in df.columns if c.endswith("_weather_main") or c.endswith("_weather_desc") or c.endswith("_weather_icon")]
    numeric_df = df.drop(columns=weather_cols, errors="ignore").copy()

    for col in ["la_prcp", "sf_prcp", "sd_prcp", "sj_prcp", "fresno_prcp"]:
        if col in numeric_df.columns:
            numeric_df[col] = numeric_df[col].fillna(0)

    daily_df = (
        numeric_df.resample("D", on="datetime")
        .agg({
            "la_temp": ["mean", "max", "min"],
            "la_rhum": ["mean"],
            "la_prcp": ["sum"],
            "la_wspd": ["mean"],
            "sf_temp": ["mean", "max", "min"],
            "sf_rhum": ["mean"],
            "sf_prcp": ["sum"],
            "sf_wspd": ["mean"],
            "sd_temp": ["mean", "max", "min"],
            "sd_rhum": ["mean"],
            "sd_prcp": ["sum"],
            "sd_wspd": ["mean"],
            "sj_temp": ["mean", "max", "min"],
            "sj_rhum": ["mean"],
            "sj_prcp": ["sum"],
            "sj_wspd": ["mean"],
            "fresno_temp": ["mean", "max", "min"],
            "fresno_rhum": ["mean"],
            "fresno_prcp": ["sum"],
            "fresno_wspd": ["mean"],
        })
    )

    daily_df.columns = [f"{col[0]}_{col[1]}" for col in daily_df.columns]
    daily_df = daily_df.reset_index()

    # representative icon/description using LA midday forecast when available
    icon_rows = df[df["datetime"].dt.hour == 12].copy()
    if icon_rows.empty:
        icon_rows = df.groupby(df["datetime"].dt.normalize(), as_index=False).first()
    else:
        icon_rows = icon_rows.groupby(icon_rows["datetime"].dt.normalize(), as_index=False).first()

    icon_map = icon_rows[["datetime", f"{DISPLAY_CITY_FOR_ICON}_weather_main", f"{DISPLAY_CITY_FOR_ICON}_weather_desc", f"{DISPLAY_CITY_FOR_ICON}_weather_icon"]].copy()
    icon_map = icon_map.rename(columns={
        "datetime": "date",
        f"{DISPLAY_CITY_FOR_ICON}_weather_main": "weather_main",
        f"{DISPLAY_CITY_FOR_ICON}_weather_desc": "weather_desc",
        f"{DISPLAY_CITY_FOR_ICON}_weather_icon": "weather_icon",
    })
    icon_map["date"] = pd.to_datetime(icon_map["date"]).dt.normalize()

    daily_df = daily_df.merge(icon_map, left_on="datetime", right_on="date", how="left")
    daily_df = daily_df.drop(columns=["date"], errors="ignore")

    daily_df["year"] = daily_df["datetime"].dt.year
    daily_df["month"] = daily_df["datetime"].dt.month
    daily_df["day"] = daily_df["datetime"].dt.day
    daily_df["day_of_week"] = daily_df["datetime"].dt.dayofweek
    daily_df["day_name"] = daily_df["datetime"].dt.day_name()
    daily_df["day_of_year"] = daily_df["datetime"].dt.dayofyear
    daily_df["is_weekend"] = daily_df["day_of_week"].isin([5, 6]).astype(int)

    return daily_df.head(4).copy()


@st.cache_data(ttl=60 * 30)
def fetch_all_city_forecasts(api_key, units="metric"):
    weather_dfs = {}
    for city_name, coords in CITY_COORDS.items():
        forecast_json = get_forecast_json(coords["lat"], coords["lon"], api_key, units=units)
        weather_dfs[city_name] = forecast_json_to_df(forecast_json)
    return weather_dfs


@st.cache_data(ttl=60 * 15)
def fetch_previous_day_load_mw_mean():
    caiso = gridstatus.CAISO()
    yesterday = (pd.Timestamp.now(tz="America/Los_Angeles") - pd.Timedelta(days=1)).date()
    load_df = caiso.get_load(yesterday)

    load_col = None
    for candidate in ["Load", "load"]:
        if candidate in load_df.columns:
            load_col = candidate
            break

    if load_col is None:
        raise ValueError(f"Could not find load column in GridStatus response: {load_df.columns.tolist()}")

    return float(load_df[load_col].mean())


def add_aggregated_weather_features(df, base_temp=65, hot_temp=75, very_hot_temp=85):
    df = df.copy()

    temp_mean_cols = [col for col in df.columns if col.endswith("_temp_mean")]
    temp_max_cols = [col for col in df.columns if col.endswith("_temp_max")]
    temp_min_cols = [col for col in df.columns if col.endswith("_temp_min")]
    rhum_mean_cols = [col for col in df.columns if col.endswith("_rhum_mean")]
    prcp_sum_cols = [col for col in df.columns if col.endswith("_prcp_sum")]
    wspd_mean_cols = [col for col in df.columns if col.endswith("_wspd_mean")]

    df["temp_mean_all"] = df[temp_mean_cols].mean(axis=1)
    df["temp_max_all"] = df[temp_max_cols].mean(axis=1)
    df["temp_min_all"] = df[temp_min_cols].mean(axis=1)
    df["rhum_mean_all"] = df[rhum_mean_cols].mean(axis=1)
    df["prcp_sum_all"] = df[prcp_sum_cols].sum(axis=1)
    df["wspd_mean_all"] = df[wspd_mean_cols].mean(axis=1)

    # convert C to F because training data engineered these with base_temp=65, hot_temp=75, very_hot_temp=85
    temp_mean_all_f = df["temp_mean_all"] * 9 / 5 + 32

    df["temp_range_all"] = df["temp_max_all"] - df["temp_min_all"]
    df["cooling_degree"] = np.maximum(0, temp_mean_all_f - base_temp)
    df["heating_degree"] = np.maximum(0, base_temp - temp_mean_all_f)
    df["cooling_degree_sq"] = df["cooling_degree"] ** 2
    df["heating_degree_sq"] = df["heating_degree"] ** 2
    df["is_hot"] = (temp_mean_all_f >= hot_temp).astype(int)
    df["is_very_hot"] = (temp_mean_all_f >= very_hot_temp).astype(int)

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


def build_single_row_features(row_df, lag1_load):
    df = row_df.copy()
    df["lag1_load"] = lag1_load
    df = add_aggregated_weather_features(df)
    df = add_calendar_features(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)
    return df


def weather_emoji(desc):
    text = (desc or "").lower()
    if "thunder" in text:
        return "⛈️"
    if "snow" in text:
        return "❄️"
    if "rain" in text or "drizzle" in text:
        return "🌧️"
    if "cloud" in text:
        return "☁️"
    if "clear" in text:
        return "☀️"
    if "mist" in text or "fog" in text or "haze" in text:
        return "🌫️"
    return "🌤️"


def load_arrow(current_value, previous_value):
    if current_value > previous_value + 1:
        return "📈"
    if current_value < previous_value - 1:
        return "📉"
    return "➡️"


def main():
    st.title("California Electricity Demand Forecast")
    st.caption("4-day forecast using OpenWeatherMap weather forecasts + yesterday's CAISO load from GridStatus + your trained linear regression model.")

    if not OPENWEATHER_API_KEY:
        st.error("Missing OPENWEATHER_API_KEY. Add it to Streamlit secrets before running the app.")
        st.stop()

    if gridstatus is None:
        st.error(f"gridstatus failed to import: {GRIDSTATUS_IMPORT_ERROR}")
        st.stop()

    model = load_model()
    expected_features = get_expected_feature_columns(model)

    with st.spinner("Pulling forecast weather and CAISO load..."):
        city_weather_dfs = fetch_all_city_forecasts(OPENWEATHER_API_KEY, units="metric")
        merged_forecast_df = merge_city_forecasts(city_weather_dfs)
        daily_weather_df = build_daily_forecast_features(merged_forecast_df)
        initial_lag1_load = fetch_previous_day_load_mw_mean()

        feature_rows = []
        predictions = []
        rolling_lag = initial_lag1_load

        for _, raw_row in daily_weather_df.iterrows():
            raw_row_df = pd.DataFrame([raw_row])
            feature_row = build_single_row_features(raw_row_df, rolling_lag)
            X_one = feature_row[expected_features].copy()
            pred = float(model.predict(X_one)[0])

            feature_rows.append(feature_row)
            predictions.append(pred)
            rolling_lag = pred

        inference_df = pd.concat(feature_rows, ignore_index=True)
        X_pred = inference_df[expected_features].copy()
        preds = np.array(predictions)

    results = inference_df[["datetime", "weather_main", "weather_desc", "weather_icon", "temp_mean_all", "prcp_sum_all"]].copy()
    results["predicted_load_mw_mean"] = preds
    results["date"] = pd.to_datetime(results["datetime"]).dt.strftime("%a %b %d")

    st.subheader("Forecast cards")
    cols = st.columns(len(results))

    previous_pred = initial_lag1_load
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i]:
            arrow = load_arrow(row["predicted_load_mw_mean"], previous_pred)
            wx = weather_emoji(row["weather_desc"])
            st.markdown(f"### {row['date']}")
            st.markdown(f"{wx}  {arrow}")
            st.metric(
                label="Predicted mean load",
                value=f"{row['predicted_load_mw_mean']:,.0f} MW",
                delta=f"{row['predicted_load_mw_mean'] - previous_pred:,.0f} MW",
            )
            st.caption(f"{row['weather_desc'].title() if pd.notna(row['weather_desc']) else 'Forecast unavailable'}")
            st.caption(f"Avg temp: {row['temp_mean_all']:.1f}°C")
            st.caption(f"Precip: {row['prcp_sum_all']:.1f} mm")
            previous_pred = row["predicted_load_mw_mean"]

    st.subheader("Forecast chart")
    chart_df = results[["datetime", "predicted_load_mw_mean"]].copy().set_index("datetime")
    st.line_chart(chart_df)

    st.subheader("Forecast table")
    display_df = results[["date", "weather_desc", "temp_mean_all", "prcp_sum_all", "predicted_load_mw_mean"]].copy()
    display_df = display_df.rename(columns={
        "weather_desc": "weather",
        "temp_mean_all": "avg_temp_c",
        "prcp_sum_all": "precip_mm",
    })
    st.dataframe(display_df, use_container_width=True)

    with st.expander("Debug / model inputs"):
        st.write("Expected model feature columns:")
        st.write(expected_features)
        st.write("Daily weather features before engineering:")
        st.dataframe(daily_weather_df, use_container_width=True)
        st.write("Final inference frame sent to model:")
        st.dataframe(X_pred, use_container_width=True)
        st.write(f"Yesterday mean CAISO load used as initial lag1_load: {initial_lag1_load:,.2f} MW")

    st.info(
        "Important note: because the model uses lag1_load, the app uses a recursive forecast: day 1 uses yesterday's actual CAISO mean load, then each later day uses the prior day's prediction as its lag input."
    )


if __name__ == "__main__":
    main()
