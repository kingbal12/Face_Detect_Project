type_1_max, type_1_min = 255, 232.7
type_2_max, type_2_min = 232.6, 218.7
type_3_max, type_3_min = 218.6, 191.3
type_4_max, type_4_min = 191.2, 130.7
type_5_max, type_5_min = 130.6, 46
type_6_max, type_6_min = 45.9, 6

FOREHEAD = ""

BETWEEN_THE_EYEBROWS = ""

NOSE = ""

NOSE_TIP = ""

RIGHT_EYE_RIGHT = ""

LEFT_EYE_LEFT = ""

UNDER_THE_EYE_RIGHT = ""

UNDER_THE_EYE_LEFT = ""

RIGTH_CHEEK = ""

LEFT_CHEEK = ""

RIGHT_CORNER_OF_MOUTH = ""

LEFT_CORNER_OF_MOUTH = ""

RIGHT_CHIN = ""

LEFT_CHIN = ""

# 시술 후 이미지 열기


def GET_SCORE(skintone):
    if skintone >= type_1_min and skintone <= type_1_max:
        type = "type1"
        score = 100 - (type_1_max - skintone) / ((type_1_max - type_1_min) / 100)
    elif skintone >= type_2_min and skintone <= type_2_max:
        type = "type2"
        score = 100 - (type_2_max - skintone) / ((type_2_max - type_2_min) / 100)
    elif skintone >= type_3_min and skintone <= type_3_max:
        type = "type3"
        score = 100 - (type_3_max - skintone) / ((type_3_max - type_3_min) / 100)
    elif skintone >= type_4_min and skintone <= type_4_max:
        type = "type4"
        score = 100 - (type_4_max - skintone) / ((type_4_max - type_4_min) / 100)
    elif skintone >= type_5_min and skintone <= type_5_max:
        type = "type5"
        score = 100 - (type_5_max - skintone) / ((type_5_max - type_5_min) / 100)
    elif skintone >= type_6_min and skintone <= type_6_max:
        type = "type6"
        score = 100 - (type_6_max - skintone) / ((type_6_max - type_6_min) / 100)
    else:
        score = 0

    print("Type: ", type, "\n", "Score: ", score)
    return [type, score]


GET_SCORE(89)
