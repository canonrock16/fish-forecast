# %%
import numpy as np
import pandas as pd

# %%
# area_1148 = pd.read_csv("./catches_area_1148_from_20230527_to_20200501.csv_with_details.csv")
# area_1150 = pd.read_csv("./catches_area_1150_from_20230527_to_20200501.csv_with_details.csv")
# pd.concat([area_1148, area_1150]).reset_index(drop=True).drop_duplicates(subset=["catch_id"]).sort_values(by=["caught_at"], ascending=False).to_csv(
#     "area_1148_1150_from_20230527_to_20200501_with_details.csv", index=False
# )
df = pd.read_csv("./area_1148_1150_from_20230527_to_20200501_with_details.csv")


# %%
df

# %%
df["caught_at"] = pd.to_datetime(df["caught_at"])
df["date"] = df["caught_at"].dt.date
# %%

# yを算出
fish_count_df = (
    df.groupby(
        [
            "date",
            "fish_name",
            "area_name",
        ]
    )
    .size()
    .reset_index(name="fish_count")
)

# %%
fish_count_df

# %%

user_count_df = (
    df.groupby(["date", "user_name", "area_name", "fish_name"])
    .size()
    .groupby(["date", "area_name", "fish_name"])
    .size()
    .reset_index(name="user_count")
)
# %%
user_count_df
# %%
Y_df = pd.merge(
    fish_count_df,
    user_count_df,
    on=["date", "area_name", "fish_name"],
    how="left",
)
Y_df

# %%

Y_df["fish_per_user"] = Y_df["fish_count"] / Y_df["user_count"]
Y_df["date"] = pd.to_datetime(Y_df["date"])
Y_df
# %%


# %%
# 一日ごとのデータに変換
X_df = (
    df[["date", "area_name", "fish_name", "temperature", "wind_direction", "wind_speed", "pressure", "tide_name", "month_age", "weather"]]
    .groupby(["date", "area_name", "fish_name"])
    .agg(
        {
            "temperature": "mean",
            "wind_direction": "max",
            "wind_speed": "mean",
            "pressure": "mean",
            "tide_name": "max",
            "month_age": "max",
            "weather": "max",
        }
    )
    .reset_index()
)


# %%
# 日時に関する処理
# 年、月、日に分割
X_df["date"] = pd.to_datetime(X_df["date"])
X_df["year"] = X_df["date"].dt.year
X_df["month"] = X_df["date"].dt.month
X_df["day"] = X_df["date"].dt.day

# 1月からの距離
X_df["month_diff_from_january_squared"] = ((X_df["month"] - 1) % 12).map(lambda x: min(x, 12 - x) ** 2)

# Cyclical Encoding
# Month
X_df["month_sin"] = np.sin((X_df["month"] - 1) * (2.0 * np.pi / 12))
X_df["month_cos"] = np.cos((X_df["month"] - 1) * (2.0 * np.pi / 12))

# Day
X_df["day_sin"] = np.sin((X_df["day"] - 1) * (2.0 * np.pi / 31))
X_df["day_cos"] = np.cos((X_df["day"] - 1) * (2.0 * np.pi / 31))
# %%
X_df


# %%
input_df = pd.merge(X_df, Y_df[["date", "area_name", "fish_name", "fish_per_user"]], on=["date", "area_name", "fish_name"], how="left")
input_df


# %%

# %%


# %%
