# 結局こっちは使っていない
# %%
import time

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

anglers_area_url = "https://anglers.jp/areas/1148/catches"
# Setup Chrome options
options = Options()
options.add_argument("--headless")  # Ensure GUI is off
chrome_driver_path = "/usr/local/bin/chromedriver"
driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)
# driver = webdriver.Chrome(executable_path=chrome_driver_path)

# %%
# Navigate to the page
driver.get(anglers_area_url)

# sort by date
new_button = driver.find_element(By.XPATH, '//button[contains(text(), "新着")]')
new_button.click()
date_button = driver.find_element(By.XPATH, '//a[@class="dropdown-item" and contains(text(), "釣れた日")]')
date_button.click()

# %%
# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")
# This will store the links
dates = []
fish_names = []
places = []
links = []

# Start index for new elements each time we scroll
start_index = 0
while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load the page
    time.sleep(2)

    # クラス名が複数あり、スペースで区切られている場合はclasscssセレクターじゃないと取得できない
    divs = driver.find_elements(By.CSS_SELECTOR, ".col-6.col-md-3")

    for div in divs[start_index:]:
        # リンク
        a = div.find_element(By.TAG_NAME, "a")
        href = a.get_attribute("href")
        if "/catches/" in href:
            links.append(href)

        # Get fish name
        fish_name_element = div.find_element(By.CSS_SELECTOR, ".sc-kEYyzF.hSzqDC")
        fish_name = fish_name_element.text
        fish_names.append(fish_name)
        # Get place
        place_element = div.find_element(By.CSS_SELECTOR, ".sc-iAyFgw.jrGnvv")
        place = place_element.text
        places.append(place)
        # Get date
        date_element = div.find_element(By.CSS_SELECTOR, ".sc-eHgmQL.kCkMzQ")
        date = date_element.text
        dates.append(date)

        # print(f"Fish Name: {fish_name}, Place: {place}, Date: {date}, URL: {href}")
    # Update the start index to the current number of elements
    start_index = len(divs)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# Be sure to close the driver when done
driver.quit()


# %%
