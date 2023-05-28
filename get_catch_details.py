# %%
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

CATCH_CSV_PATH = "./catches_area_1150_from_20230527_to_20200501.csv"


# %%
def get_nature_info(soup):
    content_blocks = soup.find_all("div", {"class": "content-block"})
    for block in content_blocks:
        h3_tag = block.find("h3")
        if h3_tag and h3_tag.text == "状況":
            dt_elements = block.find_all("dt")
            dd_elements = block.find_all("dd")
            for dt, dd in zip(dt_elements, dd_elements):
                if dt.text.strip() == "天気":
                    weathers = dd.text.strip().split()
                    temperature = weathers[0][:-1]
                    wind_direction = weathers[1]
                    wind_speed = weathers[2][:-3]
                    pressure = weathers[3][:-3]
                elif dt.text.strip() == "潮位":
                    tide_level = dd.text.strip()[:-2]
                elif dt.text.strip() == "潮名":
                    tide_name = dd.text.strip()
                elif dt.text.strip() == "月齢":
                    month_age = dd.text.strip()

    return temperature, wind_direction, wind_speed, pressure, tide_level, tide_name, month_age


# %%
catch_ids = []
temperatures = []
wind_directions = []
wind_speeds = []
pressures = []
tide_levels = []
tide_names = []
month_ages = []

df = pd.read_csv(CATCH_CSV_PATH)
for _, data in tqdm(df.iterrows(), total=len(df)):
    catch_id = data["catch_id"]
    url = f"https://anglers.jp/catches/{catch_id}"

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        temperature, wind_direction, wind_speed, pressure, tide_level, tide_name, month_age = get_nature_info(soup)
    except:
        continue

    catch_ids.append(catch_id)
    temperatures.append(temperature)
    wind_directions.append(wind_direction)
    wind_speeds.append(wind_speed)
    pressures.append(pressure)
    tide_levels.append(tide_level)
    tide_names.append(tide_name)
    month_ages.append(month_age)
    # %%
data = {
    "catch_id": catch_ids,
    "temperature": temperatures,
    "wind_direction": wind_directions,
    "wind_speed": wind_speeds,
    "pressure": pressures,
    "tide_level": tide_levels,
    "tide_name": tide_names,
    "month_age": month_ages,
}

detail_df = pd.DataFrame(data)


# %%
detail_df
# %%
result_df = pd.merge(df, detail_df, on="catch_id", how="inner")

# %%
result_df.to_csv(f"{CATCH_CSV_PATH}_with_details.csv", index=False)

# %%
