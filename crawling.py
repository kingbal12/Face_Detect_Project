import warnings

warnings.filterwarnings(action="ignore")

import pandas as pd
import numpy as np

from tqdm import tqdm_notebook
from bs4 import BeautifulSoup
from html_table_parser import parser_functions
import requests

import re

cols = ["성분코드", "성분명", "영문명", "CAS No", "구명칭"]
ing_df = pd.DataFrame(columns=cols)


for num in range(1, 2038):  # 1~2038의 페이지로 구성되어있다
    url = f"https://kcia.or.kr/cid/search/ingd_list.php?page={num}"  # url이 뒤의 page num만 바뀜
    response = requests.get(url)

    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, "html.parser", from_encoding="utf-8")

    data = soup.find("table", {"class": "bbs_list"})  # class가 bbs_list인 테이블을 찾아와서
    data = parser_functions.make2d(data)[1:]  # parser_functions의 make2d로 받아온다

    tmpdf = pd.DataFrame(data, columns=cols)  # 임시 데이터 프레임을 만들어 각 페이지 별 테이블 정보를 담아서
    ing_df = pd.concat([ing_df, tmpdf])  # 위에 만들어 놓은 빈 데이터프레임이랑 concat 하는 식으로 누적시킨다

ing_df


set(list(range(1, 22997))) - set(list(ing_df["성분코드"].astype("int")))


ing_df = ing_df.astype({"성분코드": "int"})
ing_df = ing_df[["성분코드", "성분명", "CAS No", "구명칭", "영문명"]].set_index("성분코드").sort_index()
ing_df.reset_index(inplace=True)


pattern = r"\([^)]*\)"

for idx, row in ing_df.iterrows():
    tmp = ing_df.iloc[idx]["영문명"]
    try:
        if "(" in tmp:
            txt = re.sub(pattern=pattern, repl="", string=tmp)
            txt = " ".join(txt.split())
            ing_df.iloc[idx, 2] = txt
    except:
        pass

ing_df["formatted_영문명"] = ing_df["영문명"].str.lower().str.replace(" ", "-")
# inci-decoder에 검색가능한 format으로 변경하여 컬럼 추가

ing_df.head(10)  # 확인 한 번 해주고
ing_df.to_csv("cosmetics_data.csv")  # 저장
