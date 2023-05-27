# %%

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

# %%

AREA_ID = 1148  # 水の広場公園
# AREA_ID = 1150  # 有明西ふ頭公園
DATE_OLDER_LIMIT = "2020-05-01"  # 3年間くらい取得

headers = {
    "authority": "anglers.jp",
    "accept": "application/json, text/plain, */*",
    "accept-language": "ja,en-US;q=0.9,en;q=0.8",
    "cookie": "ahoy_visitor=78d128ab-259b-496c-846b-40f1abda6e27; _anglers_production=7abc51514441c6f2d221249648a1b977; _anglers_session_production=850b6b233262dcb0aa64304ce32819ce; ahoy_visit=8107466e-38be-4f6b-b930-e28cafadd7bc",
    "referer": "https://anglers.jp/areas/1148/catches?page=4",
    "sec-ch-ua": '"Not?A_Brand";v="8", "Chromium";v="108", "Google Chrome";v="108"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
}


catch_ids = []
caught_ats = []
user_ids = []
user_names = []
area_ids = []
area_names = []
fish_ids = []
fish_names = []
sizes = []
lures = []

page_id = 1
while True:
    url = f"https://anglers.jp/api/v2/results.json?page={page_id}&order=caught_at&area_id={AREA_ID}"
    response = requests.get(url, headers=headers)
    results = response.json()

    for result in results:
        try:
            catch_id = result["id"]
            caught_at = datetime.strptime(result["caught_at"], "%Y-%m-%dT%H:%M:%S.%f%z")
            user_id = result["user"]["id"]
            user_name = result["user"]["name"]
            area_id = result["area"]["id"]
            area_name = result["area"]["name"]
            fish_id = result["fish"]["id"]
            fish_name = result["fish"]["name"]
            size = result["size"]
            lure = result["lure_details"]

            catch_ids.append(catch_id)
            caught_ats.append(caught_at)
            user_ids.append(user_id)
            user_names.append(user_name)
            area_ids.append(area_id)
            area_names.append(area_name)
            fish_ids.append(fish_id)
            fish_names.append(fish_name)
            sizes.append(size)
            lures.append(lure)

        except KeyError:
            pass

    if caught_at < datetime.strptime(DATE_OLDER_LIMIT, "%Y-%m-%d").replace(tzinfo=timezone(timedelta(hours=9))):
        break
    page_id += 1

# %%


data = {
    "catch_id": catch_ids,
    "caught_at": caught_ats,
    "user_id": user_ids,
    "user_name": user_names,
    "area_id": area_ids,
    "area_name": area_names,
    "fish_id": fish_ids,
    "fish_name": fish_names,
    "size": sizes,
    "lure": lures,
}

df = pd.DataFrame(data)

# %%
df
# %%

today = datetime.now().strftime("%Y%m%d")
limit = datetime.strptime(DATE_OLDER_LIMIT, "%Y-%m-%d").strftime("%Y%m%d")
df.to_csv(f"catches_area_{AREA_ID}_from_{today}_to_{limit}.csv", index=False)
# %%
