# %%

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# %%
# area_1148 = pd.read_csv("./catches_area_1148_from_20230527_to_20200501.csv_with_details.csv")
# area_1150 = pd.read_csv("./catches_area_1150_from_20230527_to_20200501.csv_with_details.csv")
# pd.concat([area_1148, area_1150]).reset_index(drop=True).drop_duplicates(subset=["catch_id"]).sort_values(by=["caught_at"], ascending=False).to_csv(
#     "area_1148_1150_from_20230527_to_20200501_with_details.csv", index=False
# )
df = pd.read_csv("./area_1148_1150_from_20230527_to_20200501_with_details.csv")

# %%
df.isnull().sum()
# %%
df = df.dropna(subset=["month_age"])
# %%
df.isnull().sum()
# %%
df["weather"].unique()
# %%
fish_mapping = {
    "セイゴ（マルスズキ）": "シーバス",
    "セイゴ（タイリクスズキ）": "シーバス",
    "セイゴ（ヒラスズキ）": "シーバス",
    "スズキ": "シーバス",
    "タイリクスズキ": "シーバス",
    "フッコ（マルスズキ）": "シーバス",
    "フッコ（ヒラスズキ）": "シーバス",
    "フッコ（タイリクスズキ）": "シーバス",
    "クロダイ": "チヌ",
    "チンチン": "チヌ",
    "マハゼ": "ハゼ",
    "ウロハゼ": "ハゼ",
    "シマハゼ": "ハゼ",
    "アゴハゼ": "ハゼ",
    "クロメバル": "メバル",
    "シロメバル": "メバル",
    "マアジ": "アジ",
    "カタクチイワシ": "イワシ",
    "トウゴロウイワシ": "イワシ",
    "マサバ": "サバ",
    "ゴマサバ": "サバ",
    "アカエイ": "エイ",
    "モクズガニ": "カニ",
    "アカカマス": "カマス",
    "ムラソイ": "ソイ",
}

df["fish_name"] = df["fish_name"].replace(fish_mapping)
print(df["fish_name"].unique())

fish_to_keep = ["シーバス", "チヌ", "キビレ", "ボラ", "サッパ", "コノシロ", "ハゼ", "メバル", "イワシ", "アジ", "サバ", "カサゴ", "カレイ"]
# fish_to_keep = ["シーバス", "チヌ", "エイ", "キビレ", "ボラ", "サッパ", "コノシロ", "ハゼ", "メバル", "イワシ", "アジ", "サバ", "カサゴ", "カレイ"]
# fish_to_keep = ["シーバス", "チヌ", "エイ", "キビレ", "アイナメ", "ボラ", "サッパ", "コノシロ", "ハゼ", "メバル", "イワシ", "カマス", "アジ", "サバ", "タチウオ", "ソイ", "カサゴ", "サヨリ", "カレイ"]
df = df[df["fish_name"].isin(fish_to_keep)]

# %%
df["caught_at"] = pd.to_datetime(df["caught_at"])
df["date"] = df["caught_at"].dt.date
# %%

# yを算出
if True:
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
    user_count_df = (
        df.groupby(["date", "user_name", "area_name", "fish_name"])
        .size()
        .groupby(["date", "area_name", "fish_name"])
        .size()
        .reset_index(name="user_count")
    )
    Y_df = pd.merge(
        fish_count_df,
        user_count_df,
        on=["date", "area_name", "fish_name"],
        how="left",
    )
if False:
    fish_count_df = (
        df.groupby(
            [
                "date",
                "area_name",
            ]
        )
        .size()
        .reset_index(name="fish_count")
    )
    user_count_df = df.groupby(["date", "user_name", "area_name"]).size().groupby(["date", "area_name"]).size().reset_index(name="user_count")
    Y_df = pd.merge(
        fish_count_df,
        user_count_df,
        on=["date", "area_name"],
        how="left",
    )

Y_df["fish_per_user"] = Y_df["fish_count"] / Y_df["user_count"]
Y_df["date"] = pd.to_datetime(Y_df["date"])


# %%
# 一日ごとのデータに変換
X_df = (
    df[["date", "area_name", "fish_name", "temperature", "wind_direction", "wind_speed", "pressure", "tide_name", "month_age", "weather"]]
    # df[["date", "area_name", "temperature", "wind_direction", "wind_speed", "pressure", "tide_name", "month_age", "weather"]]
    .groupby(["date", "area_name", "fish_name"])
    # .groupby(["date", "area_name"])
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
    ).reset_index()
)

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

# month_age
X_df["month_age_sin"] = np.sin((X_df["month_age"] - 1) * (2.0 * np.pi / 29))
X_df["month_age_cos"] = np.cos((X_df["month_age"] - 1) * (2.0 * np.pi / 29))
X_df["month_diff_from_29_squared"] = ((X_df["month"] - 1) % 29).map(lambda x: min(x, 29 - x) ** 2)

# %%
input_df = pd.merge(X_df, Y_df[["date", "area_name", "fish_name", "fish_per_user"]], on=["date", "area_name", "fish_name"], how="left")
# input_df = pd.merge(X_df, Y_df[["date", "area_name", "fish_per_user"]], on=["date", "area_name"], how="left")
input_df = input_df.drop(columns=["date"])
input_df = input_df[
    [
        "area_name",
        "fish_name",
        "temperature",
        # "wind_direction",
        "wind_speed",
        "pressure",
        "tide_name",
        "month_age",
        "weather",
        # "year",
        "month",
        "day",
        # "month_diff_from_january_squared",
        "month_sin",
        # "month_cos",
        "day_sin",
        # "day_cos",
        "month_age_sin",
        # "month_age_cos",
        # "month_diff_from_29_squared",
        "fish_per_user",
    ]
]

# categorical_columns = ["area_name", "fish_name", "wind_direction", "tide_name", "weather"]
categorical_columns = ["area_name", "fish_name", "tide_name", "weather"]
encoders = {col: LabelEncoder() for col in categorical_columns}
for col, encoder in encoders.items():
    input_df[col] = encoder.fit_transform(input_df[col])
input_df

# %%
X_train, X_test, y_train, y_test = train_test_split(
    input_df.drop(columns=["fish_per_user"]), input_df["fish_per_user"], test_size=0.3, random_state=42
)
# %%
if True:
    clf = DecisionTreeRegressor(random_state=42, max_leaf_nodes=10, min_samples_leaf=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("The rmse of prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

    importances = clf.feature_importances_
    feature_importance = dict(zip(X_train.columns, importances))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for i in sorted_features:
        print("Feature: %s, Score: %.5f" % (i[0], i[1]))

    plt.figure(figsize=(20, 20))  # set the figure size
    plot_tree(clf, filled=True, feature_names=input_df.drop(columns=["fish_per_user"]).columns, class_names=["Class 0", "Class 1"])
    plt.show()
# %%
if True:
    lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_columns)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train, categorical_feature=categorical_columns)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": {"l2", "l1"},
        "num_leaves": 31,
        "learning_rate": 0.05,
        # "learning_rate": 0.01,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 0,
    }
    evals_result = {}
    print("Starting training...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_train, lgb_eval],
        valid_names=["Train", "Eval"],
        evals_result=evals_result,
        early_stopping_rounds=10,
    )
    print("Printing evaluation results...")

    # plot learning curves
    plt.figure(figsize=(12, 5))
    for i in ["Train", "Eval"]:
        # plt.plot(evals_result[i]["l2"], label=i)
        rmse_values = np.sqrt(evals_result[i]["l2"])  # calculate RMSE from MSE
        plt.plot(rmse_values, label=i)
    plt.legend()
    plt.xlabel("Boosting round")
    plt.ylabel("Root Mean squared error")
    plt.title("Learning curves")
    plt.show()

    print("Starting predicting...")
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    print("The rmse of prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

    # Print the name and importance of each feature
    importances = gbm.feature_importance()
    feature_names = gbm.feature_name()
    importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    sorted_importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    for key, value in sorted_importance_dict.items():
        print(f"{key}: {value}")

    print("Saving model...")
    gbm.save_model("model.txt")


# %%
if False:
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)

    print("Starting predicting...")
    y_pred = rf.predict(X_test)
    print("The rmse of prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

    # Print the name and importance of each feature
    importances = rf.feature_importances_
    feature_names = X_train.columns
    importance_dict = {name: importance for name, importance in zip(feature_names, importances)}
    sorted_importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    for key, value in sorted_importance_dict.items():
        print(f"{key}: {value}")

    # Select one tree from the forest for visualization
    estimator = rf.estimators_[5]

    # Visualize the tree
    plt.figure(figsize=(150, 100))
    tree.plot_tree(estimator, filled=True, feature_names=input_df.columns, rounded=True)
    plt.show()

    # %%

input_df.to_csv("input_df.csv", index=False)
# %%
print(encoders["weather"].classes_)
# %%
# 未来の予測をしてみる
# https://www.surf-life.blue/weather/forecast/%E6%9D%B1%E4%BA%AC%E9%83%BD/%E6%B0%B4%E3%81%AE%E5%BA%83%E5%A0%B4%E5%85%AC%E5%9C%92/
# https://saltclip.net/areas/tokyo/weathers/%E6%B0%B4%E3%81%AE%E5%BA%83%E5%A0%B4%E5%85%AC%E5%9C%92
predict_df = pd.DataFrame(
    {
        "area_name": [0, 0, 0, 0, 0],
        "fish_name": [8, 8, 8, 8, 8],
        "temperature": [20.5, 19, 19.5, 21, 21.5],
        # "wind_direction":[0,0,0,0,0,0,0],
        "wind_speed": [2, 2, 3, 8, 6],
        "pressure": [1009, 1012, 1011, 1002, 1007],
        "tide_name": ["長潮", "若潮", "中潮", "中潮", "大潮"],
        "month_age": [10.5, 11.5, 12.5, 13.5, 14.5],
        "weather": ["小雨", "曇り", "晴れ", "曇り", "小雨"],
        "year": [2023, 2023, 2023, 2023, 2023],
        "month": [5, 5, 6, 6, 6],
        "day": [30, 31, 1, 2, 3],
        # "month_diff_from_january_squared",
        "month_sin": [0.8660254037844388, 0.8660254037844388, 0.5000000000000003, 0.5000000000000003, 0.5000000000000003],
        # "month_cos",
        "day_sin": [-0.3943558551133187, -0.20129852008866028, 0.0, 0.20129852008866006, 0.39435585511331855],
        # "day_cos",
        "month_age_sin": [0.8835120444460229, 0.7621620551276365, 0.6051742151937651, 0.41988910156026443, 0.21497044021102427],
        # "month_age_cos",
        # "month_diff_from_29_squared",
        # "fish_per_user",
    }
)
for col, encoder in encoders.items():
    predict_df[col] = encoder.fit_transform(predict_df[col])

# %%
predict_df
# %%
# lgb_predict = lgb.Dataset(X_train, categorical_feature=categorical_columns)
gbm.predict(predict_df, num_iteration=gbm.best_iteration)
# %%
np.sin((14.5 - 1) * (2.0 * np.pi / 29))
# %%
np.sin((6 - 1) * (2.0 * np.pi / 12))
# %%
np.sin((3 - 1) * (2.0 * np.pi / 31))
# %%
y_pred

# %%
