# %%

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
input_df = input_df.drop(columns=["date"])


# %%

categorical_columns = ["area_name", "fish_name", "wind_direction", "tide_name", "weather"]
# Convert categorical columns to 'category' data type
# for col in categorical_columns:
# input_df[col] = input_df[col].astype("category")

# Create a LabelEncoder for each categorical column
encoders = {col: LabelEncoder() for col in categorical_columns}
# fit and transform each encoder to the corresponding column
for col, encoder in encoders.items():
    input_df[col] = encoder.fit_transform(input_df[col])

# %%
input_df
# %%
# create a train/test split
X_train, X_test, y_train, y_test = train_test_split(
    input_df.drop(columns=["fish_per_user"]), input_df["fish_per_user"], test_size=0.2, random_state=42
)
# %%

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_columns)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=categorical_columns)
# %%

# specify your configurations as a dict
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": {"l2", "l1"},
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}

print("Starting training...")
# train
gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)

print("Saving model...")
# save model to file
gbm.save_model("model.txt")

print("Starting predicting...")
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print("The rmse of prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

# %%
