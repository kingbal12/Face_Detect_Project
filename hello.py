from flask import Flask, jsonify, escape, request
import cv2
import numpy as np
import urllib.request
import json
from wrinkle_analysis import measure_wrinkle, make_landmark_points
from landmark_rgb import find_rgb
from landmark_rgb_score import rgb_score
from rgb_score_hundred import rgb_score_p
from rgb_score_s import rgb_score_s
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 이미지 서버 url
serverurl = "https://health.iot4health.co.kr:8300/"


# url 경로상 이미지 확인, 타입 변경
def url_to_image(url):
    try:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image
    except:
        return "error"


# 표정 사진 5장의 주름 점수 출력
def wrinkleFive(photo):
    aurl = serverurl + photo["A"]
    burl = serverurl + photo["B"]
    gurl = serverurl + photo["G"]
    surl = serverurl + photo["S"]
    uurl = serverurl + photo["U"]

    global errarray
    global errarrayresult
    errarray = []
    errarrayresult = []

    photoa = url_to_image(aurl)
    photob = url_to_image(burl)
    photog = url_to_image(gurl)
    photos = url_to_image(surl)
    photou = url_to_image(uurl)

    if photoa == "error":
        errarray.append("A")
    if photob == "error":
        errarray.append("B")
    if photog == "error":
        errarray.append("G")
    if photos == "error":
        errarray.append("S")
    if photou == "error":
        errarray.append("U")

    aref = measure_wrinkle(photoa)
    bref = measure_wrinkle(photob)
    gref = measure_wrinkle(photog)
    sref = measure_wrinkle(photos)
    uref = measure_wrinkle(photou)

    if aref == "error":
        errarray.append("A")
    if bref == "error":
        errarray.append("B")
    if gref == "error":
        errarray.append("G")
    if sref == "error":
        errarray.append("S")
    if uref == "error":
        errarray.append("U")

    errarrayresult = list(set(errarray))

    # 얼굴 부위별 list 변수 선언
    foreheadarray, glabellaarray, upnosearray, right_eye_sidearray, \
    left_eye_sidearray, right_eye_underarray, left_eye_underarray, \
    right_cheekarray, left_cheekarray, r_philtrumarray, l_philtrumarray, \
    r_smile_linearray, l_smile_linearray, left_moutharray, right_moutharray = ([] for _ in range(15))


    # 얼굴 부위별 주름점수 변수 선언
    scores = {}
    locations = ["foreheads", "foreheadd", "glabellas", "glabellad", "upnoses", "upnosed",
                "right_eye_sidess", "right_eye_sided", "left_eye_sidess", "left_eye_sided",
                "right_eye_underss", "right_eye_underd", "left_eye_underss", "left_eye_underd",
                "right_cheekss", "right_cheekd", "left_cheekss", "left_cheekd",
                "philtrumss", "philtrumd", "r_smile_liness", "r_smile_lined", "l_smile_liness",
                "l_smile_lined", "left_mouthss", "left_mouthd", "right_mouthss", "right_mouthd"]

    for location in locations:
        scores[f"{location}score"] = "0"



    foreheadsscore = sref["crop_forehead"]
    glabellasscore = sref["crop_glabella"]
    upnosesscore = sref["crop_upnose"]
    right_eye_sidesscore = sref["crop_right_eye_side"]
    left_eye_sidesscore = sref["crop_left_eye_side"]
    right_eye_undersscore = sref["crop_right_eye_under"]
    left_eye_undersscore = sref["crop_left_eye_under"]
    right_cheeksscore = sref["crop_right_cheek"]
    left_cheeksscore = sref["crop_left_cheek"]
    philtrumsscore = (
        float(sref["crop_r_philtrum"]) + float(sref["crop_l_philtrum"])
    ) / 2
    r_smile_linesscore = sref["crop_r_smile_line"]
    l_smile_linesscore = sref["crop_l_smile_line"]
    left_mouthsscore = sref["crop_left_mouth"]
    right_mouthsscore = sref["crop_right_mouth"]

    foreheadarray.extend(
        [
            aref["crop_forehead"],
            bref["crop_forehead"],
            gref["crop_forehead"],
            uref["crop_forehead"],
        ]
    )
    glabellaarray.extend(
        [
            aref["crop_glabella"],
            bref["crop_glabella"],
            gref["crop_glabella"],
            uref["crop_glabella"],
        ]
    )
    upnosearray.extend(
        [
            aref["crop_upnose"],
            bref["crop_upnose"],
            gref["crop_upnose"],
            uref["crop_upnose"],
        ]
    )
    right_eye_sidearray.extend(
        [
            aref["crop_right_eye_side"],
            bref["crop_right_eye_side"],
            gref["crop_right_eye_side"],
            uref["crop_right_eye_side"],
        ]
    )
    left_eye_sidearray.extend(
        [
            aref["crop_left_eye_side"],
            bref["crop_left_eye_side"],
            gref["crop_left_eye_side"],
            uref["crop_left_eye_side"],
        ]
    )
    right_eye_underarray.extend(
        [
            aref["crop_right_eye_under"],
            bref["crop_right_eye_under"],
            gref["crop_right_eye_under"],
            uref["crop_right_eye_under"],
        ]
    )
    left_eye_underarray.extend(
        [
            aref["crop_left_eye_under"],
            bref["crop_left_eye_under"],
            gref["crop_left_eye_under"],
            uref["crop_left_eye_under"],
        ]
    )
    right_cheekarray.extend(
        [
            aref["crop_right_cheek"],
            bref["crop_right_cheek"],
            gref["crop_right_cheek"],
            uref["crop_right_cheek"],
        ]
    )
    left_cheekarray.extend(
        [
            aref["crop_left_cheek"],
            bref["crop_left_cheek"],
            gref["crop_left_cheek"],
            uref["crop_left_cheek"],
        ]
    )
    r_philtrumarray.extend(
        [
            aref["crop_r_philtrum"],
            bref["crop_r_philtrum"],
            gref["crop_r_philtrum"],
            uref["crop_r_philtrum"],
        ]
    )
    l_philtrumarray.extend(
        [
            aref["crop_l_philtrum"],
            bref["crop_l_philtrum"],
            gref["crop_l_philtrum"],
            uref["crop_l_philtrum"],
        ]
    )
    r_smile_linearray.extend(
        [
            aref["crop_r_smile_line"],
            bref["crop_r_smile_line"],
            gref["crop_r_smile_line"],
            uref["crop_r_smile_line"],
        ]
    )
    l_smile_linearray.extend(
        [
            aref["crop_l_smile_line"],
            bref["crop_l_smile_line"],
            gref["crop_l_smile_line"],
            uref["crop_l_smile_line"],
        ]
    )
    left_moutharray.extend(
        [
            aref["crop_left_mouth"],
            bref["crop_left_mouth"],
            gref["crop_left_mouth"],
            uref["crop_left_mouth"],
        ]
    )
    right_moutharray.extend(
        [
            aref["crop_right_mouth"],
            bref["crop_right_mouth"],
            gref["crop_right_mouth"],
            uref["crop_right_mouth"],
        ]
    )

    foreheaddscore = max(foreheadarray)
    glabelladscore = max(glabellaarray)
    upnosedscore = max(upnosearray)
    right_eye_sidedscore = max(right_eye_sidearray)
    left_eye_sidedscore = max(left_eye_sidearray)
    right_eye_underdscore = max(right_eye_underarray)
    left_eye_underdscore = max(left_eye_underarray)
    right_cheekdscore = max(right_cheekarray)
    left_cheekdscore = max(left_cheekarray)
    philtrumdscore = (float(max(r_philtrumarray)) + float(max(l_philtrumarray))) / 2
    r_smile_linedscore = max(r_smile_linearray)
    l_smile_linedscore = max(l_smile_linearray)
    left_mouthdscore = max(left_moutharray)
    right_mouthdscore = max(right_moutharray)

    # json 구획
    foreheads = {"W_CODE": "T11C", "DIVISION": "S", "SCORE1": foreheadsscore} 
    foreheadd = {"W_CODE": "T11C", "DIVISION": "D", "SCORE1": foreheaddscore}
    glabellas = {"W_CODE": "T21C", "DIVISION": "S", "SCORE1": glabellasscore} 
    glabellad = {"W_CODE": "T21C", "DIVISION": "D", "SCORE1": glabelladscore}
    upnoses = {"W_CODE": "T31C", "DIVISION": "S", "SCORE1": upnosesscore}
    upnosed = {"W_CODE": "T31C", "DIVISION": "D", "SCORE1": upnosedscore}
    right_eye_sides = {
        "W_CODE": "E11R",
        "DIVISION": "S",
        "SCORE1": right_eye_sidesscore,
    }
    right_eye_sided =  {
        "W_CODE": "E11R",
        "DIVISION": "D",
        "SCORE1": right_eye_sidedscore,
    }
    left_eye_sides = {"W_CODE": "E11L", "DIVISION": "S", "SCORE1": left_eye_sidesscore}
    left_eye_sided = {"W_CODE": "E11L", "DIVISION": "D", "SCORE1": left_eye_sidedscore}
    right_eye_unders =  {
        "W_CODE": "E21R",
        "DIVISION": "S",
        "SCORE1": right_eye_undersscore,
    }
    right_eye_underd = {
        "W_CODE": "E21R",
        "DIVISION": "D",
        "SCORE1": right_eye_underdscore,
    }
    left_eye_unders =  {
        "W_CODE": "E21L",
        "DIVISION": "S",
        "SCORE1": left_eye_undersscore,
    }
    left_eye_underd = {
        "W_CODE": "E21L",
        "DIVISION": "D",
        "SCORE1": left_eye_underdscore,
    }
    right_cheeks = {"W_CODE": "C11R", "DIVISION": "S", "SCORE1": right_cheeksscore}
    right_cheekd = {"W_CODE": "C11R", "DIVISION": "D", "SCORE1": right_cheekdscore}
    left_cheeks = {"W_CODE": "C11L", "DIVISION": "S", "SCORE1": left_cheeksscore}
    left_cheekd = {"W_CODE": "C11L", "DIVISION": "D", "SCORE1": left_cheekdscore}
    philtrums = {"W_CODE": "U21C", "DIVISION": "S", "SCORE1": philtrumsscore}
    philtrumd = {"W_CODE": "U21C", "DIVISION": "D", "SCORE1": philtrumdscore}
    r_smile_lines = {"W_CODE": "C21R", "DIVISION": "S", "SCORE1": r_smile_linesscore}
    r_smile_lined = {"W_CODE": "C21R", "DIVISION": "D", "SCORE1": r_smile_linedscore}
    l_smile_lines = {"W_CODE": "C21L", "DIVISION": "S", "SCORE1": l_smile_linesscore}
    l_smile_lined = {"W_CODE": "C21L", "DIVISION": "D", "SCORE1": l_smile_linedscore}
    left_mouths = {"W_CODE": "U41L", "DIVISION": "S", "SCORE1": left_mouthsscore}
    left_mouthd = {"W_CODE": "U41L", "DIVISION": "D", "SCORE1": left_mouthdscore}
    right_mouths = {"W_CODE": "U41R", "DIVISION": "S", "SCORE1": right_mouthsscore}
    right_mouthd = {"W_CODE": "U41R", "DIVISION": "D", "SCORE1": right_mouthdscore}

    global wrinkleresponse

    wrinkleresponse = jsonify(
        {
            "errcode": [],
            "data": [
                foreheads,
                foreheadd,
                glabellas,
                glabellad,
                upnoses,
                upnosed,
                right_eye_sides,
                right_eye_sided,
                left_eye_sides,
                left_eye_sided,
                right_eye_unders,
                right_eye_underd,
                left_eye_unders,
                left_eye_underd,
                right_cheeks,
                right_cheekd,
                left_cheeks,
                left_cheekd,
                philtrums,
                philtrumd,
                r_smile_lines,
                r_smile_lined,
                l_smile_lines,
                l_smile_lined,
                left_mouths,
                left_mouthd,
                right_mouths,
                right_mouthd,
            ],
        }
    )

    response = wrinkleresponse

    return response, 200


# 표정 사진 3장 입력시 사용 (현재 주름미인 체크업 촬영에서 사용중)
def wrinkleThree(photo):
    gurl = serverurl + photo["G"]
    surl = serverurl + photo["S"]
    uurl = serverurl + photo["U"]

    global errarray
    global errarrayresult
    errarray = []
    errarrayresult = []

    photog = url_to_image(gurl)
    photos = url_to_image(surl)
    photou = url_to_image(uurl)

    if photog == "error":
        errarray.append("G")
    if photos == "error":
        errarray.append("S")
    if photou == "error":
        errarray.append("U")

    gref = measure_wrinkle(photog)
    sref = measure_wrinkle(photos)
    uref = measure_wrinkle(photou)

    if gref == "error":
        errarray.append("G")
    if sref == "error":
        errarray.append("S")
    if uref == "error":
        errarray.append("U")

    errarrayresult = list(set(errarray))

    # 얼굴 부위별 list 변수 선언
    foreheadarray, glabellaarray, upnosearray, right_eye_sidearray, \
    left_eye_sidearray, right_eye_underarray, left_eye_underarray, \
    right_cheekarray, left_cheekarray, r_philtrumarray, l_philtrumarray, \
    r_smile_linearray, l_smile_linearray, left_moutharray, right_moutharray = ([] for _ in range(15))

    # 얼굴 부위별 주름점수 변수 선언
    scores = {}
    locations = ["foreheads", "foreheadd", "glabellas", "glabellad", "upnoses", "upnosed",
                "right_eye_sidess", "right_eye_sided", "left_eye_sidess", "left_eye_sided",
                "right_eye_underss", "right_eye_underd", "left_eye_underss", "left_eye_underd",
                "right_cheekss", "right_cheekd", "left_cheekss", "left_cheekd",
                "philtrumss", "philtrumd", "r_smile_liness", "r_smile_lined", "l_smile_liness",
                "l_smile_lined", "left_mouthss", "left_mouthd", "right_mouthss", "right_mouthd"]

    for location in locations:
        scores[f"{location}score"] = "0"



    foreheadsscore = sref["crop_forehead"]
    glabellasscore = sref["crop_glabella"]
    upnosesscore = sref["crop_upnose"]
    right_eye_sidesscore = sref["crop_right_eye_side"]
    left_eye_sidesscore = sref["crop_left_eye_side"]
    right_eye_undersscore = sref["crop_right_eye_under"]
    left_eye_undersscore = sref["crop_left_eye_under"]
    right_cheeksscore = sref["crop_right_cheek"]
    left_cheeksscore = sref["crop_left_cheek"]
    philtrumsscore = (
        float(sref["crop_r_philtrum"]) + float(sref["crop_l_philtrum"])
    ) / 2
    r_smile_linesscore = sref["crop_r_smile_line"]
    l_smile_linesscore = sref["crop_l_smile_line"]
    left_mouthsscore = sref["crop_left_mouth"]
    right_mouthsscore = sref["crop_right_mouth"]

    foreheadarray.extend(
        [
            gref["crop_forehead"],
            uref["crop_forehead"],
        ]
    )
    glabellaarray.extend(
        [
            gref["crop_glabella"],
            uref["crop_glabella"],
        ]
    )
    upnosearray.extend(
        [
            gref["crop_upnose"],
            uref["crop_upnose"],
        ]
    )
    right_eye_sidearray.extend(
        [
            gref["crop_right_eye_side"],
            uref["crop_right_eye_side"],
        ]
    )
    left_eye_sidearray.extend(
        [
            gref["crop_left_eye_side"],
            uref["crop_left_eye_side"],
        ]
    )
    right_eye_underarray.extend(
        [
            gref["crop_right_eye_under"],
            uref["crop_right_eye_under"],
        ]
    )
    left_eye_underarray.extend(
        [
            gref["crop_left_eye_under"],
            uref["crop_left_eye_under"],
        ]
    )
    right_cheekarray.extend(
        [
            gref["crop_right_cheek"],
            uref["crop_right_cheek"],
        ]
    )
    left_cheekarray.extend(
        [
            gref["crop_left_cheek"],
            uref["crop_left_cheek"],
        ]
    )
    r_philtrumarray.extend(
        [
            gref["crop_r_philtrum"],
            uref["crop_r_philtrum"],
        ]
    )
    l_philtrumarray.extend(
        [
            gref["crop_l_philtrum"],
            uref["crop_l_philtrum"],
        ]
    )
    r_smile_linearray.extend(
        [
            gref["crop_r_smile_line"],
            uref["crop_r_smile_line"],
        ]
    )
    l_smile_linearray.extend(
        [
            gref["crop_l_smile_line"],
            uref["crop_l_smile_line"],
        ]
    )
    left_moutharray.extend(
        [
            gref["crop_left_mouth"],
            uref["crop_left_mouth"],
        ]
    )
    right_moutharray.extend(
        [
            gref["crop_right_mouth"],
            uref["crop_right_mouth"],
        ]
    )

    foreheaddscore = max(foreheadarray)
    glabelladscore = max(glabellaarray)
    upnosedscore = max(upnosearray)
    right_eye_sidedscore = max(right_eye_sidearray)
    left_eye_sidedscore = max(left_eye_sidearray)
    right_eye_underdscore = max(right_eye_underarray)
    left_eye_underdscore = max(left_eye_underarray)
    right_cheekdscore = max(right_cheekarray)
    left_cheekdscore = max(left_cheekarray)
    philtrumdscore = (float(max(r_philtrumarray)) + float(max(l_philtrumarray))) / 2
    r_smile_linedscore = max(r_smile_linearray)
    l_smile_linedscore = max(l_smile_linearray)
    left_mouthdscore = max(left_moutharray)
    right_mouthdscore = max(right_moutharray)

    # json 구획
    foreheads = {"W_CODE": "T11C", "DIVISION": "S", "SCORE1": foreheadsscore}
    foreheadd = {"W_CODE": "T11C", "DIVISION": "D", "SCORE1": foreheaddscore}
    glabellas = {"W_CODE": "T21C", "DIVISION": "S", "SCORE1": glabellasscore}
    glabellad = {"W_CODE": "T21C", "DIVISION": "D", "SCORE1": glabelladscore}
    upnoses = {"W_CODE": "T31C", "DIVISION": "S", "SCORE1": upnosesscore}
    upnosed = {"W_CODE": "T31C", "DIVISION": "D", "SCORE1": upnosedscore}
    right_eye_sides = {
        "W_CODE": "E11R",
        "DIVISION": "S",
        "SCORE1": right_eye_sidesscore,
    }
    right_eye_sided = {
        "W_CODE": "E11R",
        "DIVISION": "D",
        "SCORE1": right_eye_sidedscore,
    }
    left_eye_sides = {"W_CODE": "E11L", "DIVISION": "S", "SCORE1": left_eye_sidesscore}
    left_eye_sided = {"W_CODE": "E11L", "DIVISION": "D", "SCORE1": left_eye_sidedscore}
    right_eye_unders = {
        "W_CODE": "E21R",
        "DIVISION": "S",
        "SCORE1": right_eye_undersscore,
    }
    right_eye_underd = {
        "W_CODE": "E21R",
        "DIVISION": "D",
        "SCORE1": right_eye_underdscore,
    }
    left_eye_unders = {
        "W_CODE": "E21L",
        "DIVISION": "S",
        "SCORE1": left_eye_undersscore,
    }
    left_eye_underd = {
        "W_CODE": "E21L",
        "DIVISION": "D",
        "SCORE1": left_eye_underdscore,
    }
    right_cheeks = {"W_CODE": "C11R", "DIVISION": "S", "SCORE1": right_cheeksscore}
    right_cheekd = {"W_CODE": "C11R", "DIVISION": "D", "SCORE1": right_cheekdscore}
    left_cheeks = {"W_CODE": "C11L", "DIVISION": "S", "SCORE1": left_cheeksscore}
    left_cheekd = {"W_CODE": "C11L", "DIVISION": "D", "SCORE1": left_cheekdscore}
    philtrums = {"W_CODE": "U21C", "DIVISION": "S", "SCORE1": philtrumsscore}
    philtrumd = {"W_CODE": "U21C", "DIVISION": "D", "SCORE1": philtrumdscore}
    r_smile_lines = {"W_CODE": "C21R", "DIVISION": "S", "SCORE1": r_smile_linesscore}
    r_smile_lined = {"W_CODE": "C21R", "DIVISION": "D", "SCORE1": r_smile_linedscore}
    l_smile_lines = {"W_CODE": "C21L", "DIVISION": "S", "SCORE1": l_smile_linesscore}
    l_smile_lined = {"W_CODE": "C21L", "DIVISION": "D", "SCORE1": l_smile_linedscore}
    left_mouths = {"W_CODE": "U41L", "DIVISION": "S", "SCORE1": left_mouthsscore}
    left_mouthd = {"W_CODE": "U41L", "DIVISION": "D", "SCORE1": left_mouthdscore}
    right_mouths = {"W_CODE": "U41R", "DIVISION": "S", "SCORE1": right_mouthsscore}
    right_mouthd = {"W_CODE": "U41R", "DIVISION": "D", "SCORE1": right_mouthdscore}

    global wrinkleresponse

    wrinkleresponse = jsonify(
        {
            "errcode": [],
            "data": [
                foreheads,
                foreheadd,
                glabellas,
                glabellad,
                upnoses,
                upnosed,
                right_eye_sides,
                right_eye_sided,
                left_eye_sides,
                left_eye_sided,
                right_eye_unders,
                right_eye_underd,
                left_eye_unders,
                left_eye_underd,
                right_cheeks,
                right_cheekd,
                left_cheeks,
                left_cheekd,
                philtrums,
                philtrumd,
                r_smile_lines,
                r_smile_lined,
                l_smile_lines,
                l_smile_lined,
                left_mouths,
                left_mouthd,
                right_mouths,
                right_mouthd,
            ],
        }
    )

    response = wrinkleresponse

    return response, 200


@app.route("/")
def hello():
    name = request.args.get("name", "World")

    return f"Hello, {escape(name)}!"


# URL 변수를 사용한 사용자 요청 수신 (Test)
@app.route("/calculator/<string:operation>/<int:value1>/<int:value2>")
def calculator(operation, value1, value2):
    ret = 0
    if operation == "add":
        ret = value1 + value2
    elif operation == "sub":
        ret = value1 - value2
    elif operation == "mul":
        ret = value1 * value2
    elif operation == "div":
        ret = value1 / value2
    return f"Operation {operation} with {value1} and {value2} is {ret}"


# GET, POST 방식을 사용한 사용자 요청 수신 (Test)
@app.route("/information", methods=["GET", "POST"])
def information():
    users = [
        {"name": "John Smith", "workplace": "School", "userid": "10011"},
        {"name": "U.N. Owen", "workplace": "DoA", "userid": "10021"},
        {"name": "Guest", "workplace": "None", "userid": "10001"},
    ]
    method = ""
    if request.method == "GET":
        name = request.args.get("name", "Guest")
        workplace = request.args.get("workplace", "None")
        method = "GET"

    elif request.method == "POST":
        name = request.form["name"]
        workplace = request.form["workplace"]
        method = "POST"

    for user in users:
        if user["name"] == name and user["workplace"] == workplace:
            return f'Hello, {name}#{user["userid"]} by {method}!'

    return "Who are you?"

    # GET, POST 방식을 사용한 사용자 요청 수신


# 표정사진 갯수에 따른 분기
@app.route("/wrinkle_analysis", methods=["POST"])
def photoListLength():
    photo = request.json
    if len(photo.keys()) == 5:
        return wrinkleFive(photo)
    else:
        return wrinkleThree(photo)


# 사진 1장 분석
@app.route("/wrinkle_analysis_one", methods=["POST"])
def wrinkle_one():
    photo = request.json

    url = serverurl + photo["url"]

    errarray = []
    errarrayresult = []

    photo_one = url_to_image(url)

    if photo_one == "error":
        errarray.append("inaccurate url")

    ref = measure_wrinkle(photo_one)

    if ref == "error":
        errarray.append("inaccurate face photo")

    errarrayresult = list(set(errarray))

    foreheadsscore = ref["crop_forehead"]
    glabellasscore = ref["crop_glabella"]
    upnosesscore = ref["crop_upnose"]
    right_eye_sidesscore = ref["crop_right_eye_side"]
    left_eye_sidesscore = ref["crop_left_eye_side"]
    right_eye_undersscore = ref["crop_right_eye_under"]
    left_eye_undersscore = ref["crop_left_eye_under"]
    right_cheeksscore = ref["crop_right_cheek"]
    left_cheeksscore = ref["crop_left_cheek"]
    philtrumsscore = (float(ref["crop_r_philtrum"]) + float(ref["crop_l_philtrum"])) / 2
    r_smile_linesscore = ref["crop_r_smile_line"]
    l_smile_linesscore = ref["crop_l_smile_line"]
    left_mouthsscore = ref["crop_left_mouth"]
    right_mouthsscore = ref["crop_right_mouth"]

    # json 구획
    foreheadd = {"W_CODE": "T11C", "DIVISION": "S", "SCORE1": foreheadsscore}
    glabellad = {"W_CODE": "T21C", "DIVISION": "S", "SCORE1": glabellasscore}
    upnoses = {"W_CODE": "T31C", "DIVISION": "S", "SCORE1": upnosesscore}
    right_eye_sided = {
        "W_CODE": "E11R",
        "DIVISION": "S",
        "SCORE1": right_eye_sidesscore,
    }
    left_eye_sided = {"W_CODE": "E11L", "DIVISION": "S", "SCORE1": left_eye_sidesscore}

    right_eye_underd = {
        "W_CODE": "E21R",
        "DIVISION": "S",
        "SCORE1": right_eye_undersscore,
    }

    left_eye_underd = {
        "W_CODE": "E21L",
        "DIVISION": "S",
        "SCORE1": left_eye_undersscore,
    }

    right_cheekd = {"W_CODE": "C11R", "DIVISION": "S", "SCORE1": right_cheeksscore}
    left_cheekd = {"W_CODE": "C11L", "DIVISION": "S", "SCORE1": left_cheeksscore}
    philtrumd = {"W_CODE": "U21C", "DIVISION": "S", "SCORE1": philtrumsscore}
    r_smile_lined = {"W_CODE": "C21R", "DIVISION": "S", "SCORE1": r_smile_linesscore}
    l_smile_lined = {"W_CODE": "C21L", "DIVISION": "S", "SCORE1": l_smile_linesscore}
    left_mouthd = {"W_CODE": "U41L", "DIVISION": "S", "SCORE1": left_mouthsscore}
    right_mouthd = {"W_CODE": "U41R", "DIVISION": "S", "SCORE1": right_mouthsscore}

    wrinkleresponse = jsonify(
        {
            "errcode": [],
            "data": [
                foreheadd,
                glabellad,
                upnoses,
                right_eye_sided,
                left_eye_sided,
                right_eye_underd,
                left_eye_underd,
                right_cheekd,
                left_cheekd,
                philtrumd,
                r_smile_lined,
                l_smile_lined,
                left_mouthd,
                right_mouthd,
            ],
        }
    )

    response = wrinkleresponse

    return response, 200


@app.route("/wrinkle_analysis_one_f", methods=["POST"])
def wrinkle_one_f():
    photo = request.json

    url = photo["url"]

    errarray = []
    errarrayresult = []

    photo_one = url_to_image(url)

    if photo_one == "error":
        errarray.append("inaccurate url")

    ref = measure_wrinkle(photo_one)

    if ref == "error":
        errarray.append("inaccurate face photo")

    errarrayresult = list(set(errarray))

    foreheadsscore = ref["crop_forehead"]
    glabellasscore = ref["crop_glabella"]
    upnosesscore = ref["crop_upnose"]
    right_eye_sidesscore = ref["crop_right_eye_side"]
    left_eye_sidesscore = ref["crop_left_eye_side"]
    right_eye_undersscore = ref["crop_right_eye_under"]
    left_eye_undersscore = ref["crop_left_eye_under"]
    right_cheeksscore = ref["crop_right_cheek"]
    left_cheeksscore = ref["crop_left_cheek"]
    philtrumsscore = (float(ref["crop_r_philtrum"]) + float(ref["crop_l_philtrum"])) / 2
    r_smile_linesscore = ref["crop_r_smile_line"]
    l_smile_linesscore = ref["crop_l_smile_line"]
    left_mouthsscore = ref["crop_left_mouth"]
    right_mouthsscore = ref["crop_right_mouth"]

    # json 구획
    foreheadd = {"W_CODE": "T11C", "DIVISION": "S", "SCORE1": foreheadsscore}
    glabellad = {"W_CODE": "T21C", "DIVISION": "S", "SCORE1": glabellasscore}
    upnoses = {"W_CODE": "T31C", "DIVISION": "S", "SCORE1": upnosesscore}
    right_eye_sided = {
        "W_CODE": "E11R",
        "DIVISION": "S",
        "SCORE1": right_eye_sidesscore,
    }
    left_eye_sided = {"W_CODE": "E11L", "DIVISION": "S", "SCORE1": left_eye_sidesscore}

    right_eye_underd = {
        "W_CODE": "E21R",
        "DIVISION": "S",
        "SCORE1": right_eye_undersscore,
    }

    left_eye_underd = {
        "W_CODE": "E21L",
        "DIVISION": "S",
        "SCORE1": left_eye_undersscore,
    }

    right_cheekd = {"W_CODE": "C11R", "DIVISION": "S", "SCORE1": right_cheeksscore}
    left_cheekd = {"W_CODE": "C11L", "DIVISION": "S", "SCORE1": left_cheeksscore}
    philtrumd = {"W_CODE": "U21C", "DIVISION": "S", "SCORE1": philtrumsscore}
    r_smile_lined = {"W_CODE": "C21R", "DIVISION": "S", "SCORE1": r_smile_linesscore}
    l_smile_lined = {"W_CODE": "C21L", "DIVISION": "S", "SCORE1": l_smile_linesscore}
    left_mouthd = {"W_CODE": "U41L", "DIVISION": "S", "SCORE1": left_mouthsscore}
    right_mouthd = {"W_CODE": "U41R", "DIVISION": "S", "SCORE1": right_mouthsscore}

    wrinkleresponse = jsonify(
        {
            "errcode": [],
            "data": [
                foreheadd,
                glabellad,
                upnoses,
                right_eye_sided,
                left_eye_sided,
                right_eye_underd,
                left_eye_underd,
                right_cheekd,
                left_cheekd,
                philtrumd,
                r_smile_lined,
                l_smile_lined,
                left_mouthd,
                right_mouthd,
            ],
        }
    )

    response = wrinkleresponse

    return response, 200


@app.errorhandler(500)
def error_handling_500(error):
    return jsonify({"errcode": errarrayresult, "data": []})


@app.route("/landmark_points", methods=["POST"])
def landmarkpoints():
    photo = request.json
    photourl = serverurl + photo["photourl"]
    photurlimage = url_to_image(photourl)

    landmarkjson = make_landmark_points(photurlimage)

    response = jsonify({"data": landmarkjson})
    return response, 200


@app.route("/landmark_points_f", methods=["POST"])
def landmarkpoints_f():
    photo = request.json
    photourl = photo["photourl"]
    photurlimage = url_to_image(photourl)

    landmarkjson = make_landmark_points(photurlimage)

    response = jsonify({"data": landmarkjson})
    return response, 200


@app.route("/rgb_average", methods=["POST"])
def rgbaverage():
    rgbphoto = request.json
    rgbphotourl = serverurl + rgbphoto["rgbphotourl"]
    rgbphoturlimage = url_to_image(rgbphotourl)

    rgbjson = find_rgb(rgbphoturlimage)

    response = jsonify({"data": rgbjson})
    return response, 200


@app.route("/rgb_average_f", methods=["POST"])
def rgbaverage_f():
    rgbphoto = request.json
    rgbphotourl = rgbphoto["rgbphotourl"]
    rgbphoturlimage = url_to_image(rgbphotourl)

    rgbjson = find_rgb(rgbphoturlimage)

    response = jsonify({"data": rgbjson})
    return response, 200


@app.route("/face_part_rgb_score", methods=["POST"])
def part_rgb_score():
    rgbphoto = request.json
    rgbphotourl = serverurl + rgbphoto["rgbphotourl"]
    rgbphoturlimage = url_to_image(rgbphotourl)

    rgbjson = rgb_score(rgbphoturlimage)

    response = jsonify({"data": rgbjson})
    return response, 200


@app.route("/face_part_rgb_score_f", methods=["POST"])
def part_rgb_score_f():
    rgbphoto = request.json
    rgbphotourl = rgbphoto["rgbphotourl"]
    rgbphoturlimage = url_to_image(rgbphotourl)

    rgbjson = rgb_score(rgbphoturlimage)

    response = jsonify({"data": rgbjson})
    return response, 200


@app.route("/rgb_score_100", methods=["POST"])
def rgb_score_100():
    rgbphoto = request.json
    rgbphotourl = serverurl + rgbphoto["rgbphotourl"]
    type = rgbphoto["type"]
    rgbphoturlimage = url_to_image(rgbphotourl)

    if type == "p":
        rgbjson = rgb_score_p(rgbphoturlimage)
    elif type == "s":
        rgbjson = rgb_score_s(rgbphoturlimage)
    else:
        rgbjson = jsonify({"message": "Please enter type as s or p."})

    response = jsonify({"data": rgbjson})
    return response, 200


@app.route("/rgb_score_100_f", methods=["POST"])
def rgb_score_100_f():
    rgbphoto = request.json
    rgbphotourl = rgbphoto["rgbphotourl"]
    type = rgbphoto["type"]
    rgbphoturlimage = url_to_image(rgbphotourl)

    if type == "p":
        rgbjson = rgb_score_p(rgbphoturlimage)
    elif type == "s":
        rgbjson = rgb_score_s(rgbphoturlimage)
    else:
        rgbjson = jsonify({"message": "Please enter type as s or p."})

    response = jsonify({"data": rgbjson})
    return response, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0")
