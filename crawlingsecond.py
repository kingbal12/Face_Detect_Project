import warnings

warnings.filterwarnings(action="ignore")

import pandas as pd
import numpy as np

from tqdm import tqdm
from bs4 import BeautifulSoup
import requests
from html_table_parser import parser_functions

product_name = []  # 화장품 이름
ingredient_lst = []  # 화장품에 들어있는 성분을 리스트로 받음
formatted_ingredient_lst = []  # formatted 성분 표기명을 리스트로 받음
what_lst = []  # 성분이 어떤 효능이 있는지 리스트로 받음
failed_lst = []  # 크롤링 중 실패한 로그를 추적하기 위해

product_data = pd.read_csv("product_df.csv")


product_lst = list(product_data["product_label"].dropna())


for product in tqdm(product_lst):
    url = f"https://incidecoder.com/products/{product}"  # url에 product 명을 바꿔가며 루프를 돈다
    response = requests.get(url)

    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

    data = soup.find(
        "table", {"class": "product-skim fs16"}
    )  # class가 product-skim fs16인 테이블을 태그로 받음
    df = parser_functions.make2d(data)[1:]  # make2d를 활용해 테이블을 파이썬 자료구조로 받고
    tmpdf = pd.DataFrame(
        df, columns=["Ingredient name", "what-it-does", "irr., com.", "ID-Rating"]
    )  # 임시 데이터 프레임을 만든다

    try:  # 크롤링 시도
        # 클래스 bold인 성분들은 html 태그에 formatted_name이 존재하지 않았다.
        # 그래서 테이블에는 기록되어 있지만, 데이터를 모두 수집한 후 데이터 프레임으로 만드는 과정에서 개수가 맞지 않는 이슈가 있었다.(Ingredient_name != Formatted_name)
        # 이를 방지하기 위해 bold처리된 성분들을 없앨 것.
        Bold_lst = []
        for ing in data.find_all("td", {"class": "bold"}):
            Bold_lst.append(ing.text.replace("\n", "").strip())  # 전처리

        indexes = []
        for stop in Bold_lst:  # 볼드 처리된 성분들을 없애기 위해 인덱스를 알아내는 작업
            indexes.append(tmpdf[tmpdf["Ingredient name"] == stop].index[0])

        tmpdf.drop(
            indexes, axis=0, inplace=True
        )  # 알아온 인덱스들로 drop을 활용해 임시 데이터 프레임에서 삭제한다

        ingtmp = []
        for tag in data.find_all("a", {"class": "black ingred-detail-link"}):
            ingtmp.append(
                tag.attrs["href"][13:]
            )  # html 태그에 들어있는 formatted 성분명을 알아오는 작업. 마찬가지로 리스트 형태로 받아온다
        formatted_ingredient_lst.append(ingtmp)

        # 임시로 만든 tmpdf의 데이터들을 활용해 성분명_리스트, 효능_리스트를 얻고
        ingredient_lst.append(list(tmpdf["Ingredient name"]))
        what_lst.append(list(tmpdf["what-it-does"]))
        product_name.append(product)  # 제품명 추가

    except:  # 실패시 failed_lst에 기록
        failed_lst.append([product, data])

# 리스트로 묶여있는 원소들을 개별로 받아오는 작업
each_ingredient_lst = []
for lst in ingredient_lst:
    for ing in lst:
        each_ingredient_lst.append(ing)

each_formatted_ingredient_lst = []
for lst in formatted_ingredient_lst:
    for ing in lst:
        each_formatted_ingredient_lst.append(ing)

each_what_lst = []
for lst in what_lst:
    for does in lst:
        tmp = []
        for does in does.replace("\n", "").replace("\u200b", "").split(","):  # 전처리
            tmp.append(does.strip())
        each_what_lst.append(tmp)

cols = ["product name", "ingredients", "formatted ingredients"]
product_df = pd.DataFrame(columns=cols)
product_df["product name"] = product_name
product_df["ingredients"] = ingredient_lst


ingredient_df = pd.DataFrame(
    columns=["ingredients", "formatted ingredients", "what-it-does"]
)
ingredient_df["ingredients"] = each_ingredient_lst
ingredient_df["formatted ingredients"] = each_formatted_ingredient_lst
ingredient_df["what-it-does"] = each_what_lst

product_df.to_csv("final_product_df.csv", index=False)
ingredient_df.to_csv("ingredient_df.csv", index=False)
