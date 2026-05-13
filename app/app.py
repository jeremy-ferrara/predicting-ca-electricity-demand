import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

try:
    import gridstatus
    GRIDSTATUS_IMPORT_ERROR = None
except Exception as e:
    gridstatus = None
    GRIDSTATUS_IMPORT_ERROR = e


st.set_page_config(
    page_title="CA Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "data" / "artifacts" / "ca_electricity_demand_lr_v1.joblib"
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", os.getenv("OPENWEATHER_API_KEY", ""))

CITY_COORDS = {
    "la": {"lat": 34.0522, "lon": -118.2437},
    "sf": {"lat": 37.7749, "lon": -122.4194},
    "sd": {"lat": 32.7157, "lon": -117.1611},
    "sj": {"lat": 37.3382, "lon": -121.8863},
    "fresno": {"lat": 36.7378, "lon": -119.7871},
}

CITY_LABELS = {
    "la": "Los Angeles",
    "sf": "San Francisco",
    "sd": "San Diego",
    "sj": "San Jose",
    "fresno": "Fresno",
}

BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"


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


def build_daily_weather_summary(df, city_prefix):
    desc_col = f"{city_prefix}_weather_desc"
    main_col = f"{city_prefix}_weather_main"
    icon_col = f"{city_prefix}_weather_icon"

    icon_rows = df[df["datetime"].dt.hour == 12].copy()

    if icon_rows.empty:
        icon_rows = df.groupby(df["datetime"].dt.normalize(), as_index=False).first()
    else:
        icon_rows = icon_rows.groupby(icon_rows["datetime"].dt.normalize(), as_index=False).first()

    summary = icon_rows[["datetime", main_col, desc_col, icon_col]].copy()
    summary["date"] = pd.to_datetime(summary["datetime"]).dt.normalize()
    summary = summary.drop(columns=["datetime"])

    summary = summary.rename(columns={
        main_col: f"{city_prefix}_daily_weather_main",
        desc_col: f"{city_prefix}_daily_weather_desc",
        icon_col: f"{city_prefix}_daily_weather_icon",
    })

    return summary


def build_daily_forecast_features(model_like_df):
    df = model_like_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = filter_full_forecast_days(df)

    weather_cols = [
        c for c in df.columns
        if c.endswith("_weather_main") or c.endswith("_weather_desc") or c.endswith("_weather_icon")
    ]
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
    daily_df["date"] = pd.to_datetime(daily_df["datetime"]).dt.normalize()

    for city_prefix in CITY_COORDS.keys():
        city_summary = build_daily_weather_summary(df, city_prefix)
        daily_df = daily_df.merge(city_summary, on="date", how="left")

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

    start = (pd.Timestamp.now(tz="America/Los_Angeles") - pd.Timedelta(days=1)).normalize()
    end = start + pd.Timedelta(days=1)

    load_df = caiso.get_load(start=start, end=end)

    load_col = None
    for candidate in ["Load", "load"]:
        if candidate in load_df.columns:
            load_col = candidate
            break

    if load_col is None:
        raise ValueError("Could not find load column")

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
        return "higher than previous day"
    if current_value < previous_value - 1:
        return "lower than previous day"
    return "about the same as previous day"


def c_to_f(temp_c):
    return temp_c * 9 / 5 + 32


def main():
    st.title("⚡ California Electricity Demand Forecast")

    st.markdown(
        """
        This app estimates California's average electricity demand for the next four days.

        The forecast is generated using a linear regression machine learning model trained on historical California electricity demand, weather patterns across five major California cities, and calendar-related features.
        """
    )

    if not OPENWEATHER_API_KEY:
        st.error("Missing OpenWeather API key. Add OPENWEATHER_API_KEY to Streamlit secrets before running the app.")
        st.stop()

    if gridstatus is None:
        st.error(f"gridstatus failed to import: {GRIDSTATUS_IMPORT_ERROR}")
        st.stop()

    model = load_model()
    expected_features = get_expected_feature_columns(model)

    with st.spinner("Loading weather forecasts and recent California electricity demand..."):
        city_weather_dfs = fetch_all_city_forecasts(OPENWEATHER_API_KEY, units="metric")
        merged_forecast_df = merge_city_forecasts(city_weather_dfs)
        daily_weather_df = build_daily_forecast_features(merged_forecast_df)

        previous_day_actual_load = fetch_previous_day_load_mw_mean()

        feature_rows = []
        predictions = []
        rolling_lag = previous_day_actual_load

        for _, raw_row in daily_weather_df.iterrows():
            raw_row_df = pd.DataFrame([raw_row])
            feature_row = build_single_row_features(raw_row_df, rolling_lag)
            X_one = feature_row[expected_features].copy()
            pred = float(model.predict(X_one)[0])

            feature_rows.append(feature_row)
            predictions.append(pred)
            rolling_lag = pred

        inference_df = pd.concat(feature_rows, ignore_index=True)
        preds = np.array(predictions)

    city_weather_cols = []
    for city in CITY_COORDS.keys():
        city_weather_cols.extend([
            f"{city}_daily_weather_desc",
            f"{city}_daily_weather_icon",
        ])

    results = inference_df[
        ["datetime", "temp_mean_all", "prcp_sum_all"] + city_weather_cols
    ].copy()

    results["predicted_load_mw_mean"] = preds
    results["date"] = pd.to_datetime(results["datetime"]).dt.strftime("%a, %b %d")
    results["avg_temp_f"] = results["temp_mean_all"].apply(c_to_f)

    st.subheader("Daily forecast")

    st.caption(
        "Each card shows the model's predicted average electricity demand for that day. "
        "The weather shown below each prediction uses forecasts from Los Angeles, San Francisco, "
        "San Diego, San Jose, and Fresno."
    )

    cols = st.columns(len(results))

    previous_pred = previous_day_actual_load

    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i]:
            change_text = load_arrow(row["predicted_load_mw_mean"], previous_pred)
            change_amount = row["predicted_load_mw_mean"] - previous_pred

            st.markdown(f"### {row['date']}")
            st.metric(
                label="Predicted average demand",
                value=f"{row['predicted_load_mw_mean']:,.0f} MW",
                delta=f"{change_amount:,.0f} MW"
            )

            st.markdown("**Weather by city**")
            for city, label in CITY_LABELS.items():
                desc = row.get(f"{city}_daily_weather_desc", "")
                emoji = weather_emoji(desc)
                readable_desc = desc.title() if pd.notna(desc) and desc else "Forecast unavailable"
                st.caption(f"{emoji} {label}: {readable_desc}")

            st.caption(f"Statewide average temperature: {row['avg_temp_f']:.1f}°F")
            st.caption(f"Total precipitation estimate: {row['prcp_sum_all']:.1f} mm")

            previous_pred = row["predicted_load_mw_mean"]

    st.subheader("Forecast chart")

    st.caption("4-day forecast of predicted average California electricity demand.")

    chart_df = results[["datetime", "predicted_load_mw_mean"]].copy()

    chart_df["Date"] = pd.to_datetime(chart_df["datetime"]).dt.strftime("%a %b %d")

    line_chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Date:N",
                sort=None,
                axis=alt.Axis(labelAngle=-30, labelFontSize=12)
            ),
            y=alt.Y(
                "predicted_load_mw_mean:Q",
                title="Predicted Average Demand (MW)"
            ),
            tooltip=[
                alt.Tooltip("Date:N"),
                alt.Tooltip("predicted_load_mw_mean:Q", format=",.0f")
            ]
        )
        .properties(height=400)
    )

    st.altair_chart(line_chart, use_container_width=True)

    st.subheader("Forecast table")

    display_df = results[[
        "date",
        "avg_temp_f",
        "prcp_sum_all",
        "predicted_load_mw_mean"
    ]].copy()

    display_df = display_df.rename(columns={
        "date": "Date",
        "avg_temp_f": "Avg statewide temp (°F)",
        "prcp_sum_all": "Precipitation estimate (mm)",
        "predicted_load_mw_mean": "Predicted average demand (MW)",
    })

    display_df["Avg statewide temp (°F)"] = display_df["Avg statewide temp (°F)"].round(1)
    display_df["Precipitation estimate (mm)"] = display_df["Precipitation estimate (mm)"].round(1)
    display_df["Predicted average demand (MW)"] = display_df["Predicted average demand (MW)"].round(0).astype(int)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    with st.expander("How this forecast works"):
        st.markdown(
            """
            This app uses three main ingredients:

            1. **Weather forecasts** from several major California cities.
            2. **Recent statewide electricity demand** from CAISO, California's electric grid operator.
            3. **A trained machine learning model** that learned patterns between weather, calendar timing, and electricity use.

            For the first forecasted day, the model uses yesterday's real electricity demand.
            For the following days, it uses the previous day's prediction to continue the forecast forward.
            This keeps the app fully automated while still allowing it to produce a multi-day outlook.
            """
        )

    with st.expander("Model details"):
        st.markdown(
            """
            - Model type: Linear Regression  
            - Forecast horizon: 4 days  
            - Weather data source: OpenWeatherMap forecasts from Los Angeles, San Francisco, San Diego, San Jose, and Fresno  
            - Electricity demand data source: CAISO grid demand data accessed through GridStatus  
            - Key inputs include statewide weather conditions, recent electricity demand, seasonal patterns, and day-of-week trends  
            """
        )

        st.write(f"Previous day's actual average CAISO load: {previous_day_actual_load:,.0f} MW")

        st.write("Model features used:")
        st.write(expected_features)


if __name__ == "__main__":
    main()