from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from urllib.request import Request, urlopen

# 성분 파일(csv) 불러오기
ing_df = pd.read_csv("cosmetics_data.csv", index_col=0)
ing_df["formatted_영문명"] = ing_df["formatted_영문명"].str.replace("/", "-")


# 성분 파일에서 'formatted_영문명' 컬럼만 웹페이지 접근용으로 사용
# 결측값 제거 후 리스트로 변환
ing_list = list(ing_df["formatted_영문명"].dropna())

product_name = set()  # 제품명
product_label = set()  # 제품명 (formatted - 웹페이지 접근용)
search_failed = []  # 'formatted_영문명' 값으로 웹페이지 접근이 불가했던 건들 확인용도


# 성분으로 조회한 제품 리스트 만들기
def add_ing_products(tags):  # html tag 를 받아와 조회하여
    for tag in tags:
        if (
            tag.text not in product_name
        ):  # 중복된 데이터는 추가하지 않도록, tag의 제품명이 product_name 셋에 없는 경우에만 추가
            product_name.add(tag.text)
            product_label.add(tag.attrs["data-ga-eventlabel"][8:])
        print(product_label)


# 성분으로 접근한 웹페이지의 제품 리스트에 '다음페이지'가 존재하는지 확인하기
def next_page_exists(soup):
    if (
        "Next" in soup.find(id="product").find_all("div")[-1].text
    ):  # Next라는 문자가 해당 태그안에 존재하는지 여부 확인
        return True
    else:
        return False


for ing in tqdm(ing_list):  # 성분 리스트의 각 성분(formatted)마다
    url = "https://incidecoder.com/ingredients/" + ing  # url 주소를 생성

    # if page exists (url로 접근 가능시)
    try:
        html = urlopen(url)
        source = html.read()
        soup = BeautifulSoup(source, "html.parser")
        tags = soup.select("#product > div > a")  # html의 태그 불러오기
        add_ing_products(tags)  # 태그에서 제품명 (일반+formatted) 리스트에 저장 - 중복건은 추가 x

        if next_page_exists(soup):  # 제품리스트가 1페이지 이상인지 확인
            nextpage = True

        while nextpage:  # 다음페이지가 존재하는 경우 반복
            nexturl = soup.find(id="product").find_all("a")[-1][
                "href"
            ]  # href태그로 다음페이지 url을 생성하여 해당 페이지 접근
            url = "https://incidecoder.com" + nexturl
            html = urlopen(url)
            source = html.read()
            soup = BeautifulSoup(source, "html.parser")
            tags = soup.select("#product > div > a")
            add_ing_products(tags)  # 다음페이지에서도 동일하게 제품명 받아와서 저장

            if not next_page_exists(soup):  # 더이상 다음 페이지가 없는 경우 while문 빠져나옴
                nextpage = False

    # if page does NOT exist (url로 접근 불가시 추후 확인 용도로 search_failed 리스트에 추가)
    except Exception:
        search_failed.append(ing)
        pass

product_all = pd.DataFrame(columns=["product_label"])
product_all["product_label"] = list(product_label)
product_all.to_csv("product_df.csv", index=False)
