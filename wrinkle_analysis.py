# -*- encoding: utf8 -*-
import os
import cv2
import mediapipe as mp
import numpy as np
from skimage.filters import frangi, sato
import matplotlib.pyplot as plt
import copy
import json
import urllib.request
import traceback


# 랜드마크 포인트 json화 함수
def make_landmark_points(image):
    try:
        mp_face_mesh = mp.solutions.face_mesh

        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.multi_face_landmarks:
                raise ValueError("Face landmarks not found in the image.")
            id = 0
            landmark_points = []
            for landmark in results.multi_face_landmarks[0].landmark:
                landmark_points.append(
                    {"id": id, "x": landmark.x, "y": landmark.y, "z": landmark.z}
                )
                id = id + 1
        landmarkpoint_json = json.dumps(landmark_points).replace('"', "")
        return landmarkpoint_json
    except:
        traceback.print_exc()
        return "error"


# 원본
# eyes and nose contour
# FACEMESH_EYES_AND_NOSE = frozenset(
#     [(6, 193), (193, 55), (55, 107), (107, 66), (66, 105),
#      (105, 63), (63, 70), (70, 124), (124, 226), (226, 110),
#      (110, 24), (24, 23), (23, 22), (22, 26), (26, 233), (233, 128), # right eye region
#      (128, 114), (114, 217), (217, 198), (198, 209), (209, 49), (49, 129),
#      (129, 165), (165, 92), (92, 186), (186, 57), (57, 43), (43, 106), (106, 182), (182, 83), (83, 18),
#      (18, 313), (313, 406), (406, 335), (335, 273), (273, 287),
#      (287, 410), (410, 322), (322, 391), (391, 358),
#      (358, 279), (279, 429), (429, 420), (420, 437), (437, 343), (343, 357), # nose region
#      (357, 453), (453, 256), (256, 252), (252, 253), (253, 254),
#      (254, 339), (339, 446), (446, 353), (353, 300),
#      (300, 293), (293, 334), (334, 296), (296, 336), (336, 285),
#      (285, 417), (417, 6)] # left eye region
# )

# eyes and nose contour
FACEMESH_EYES_AND_NOSE = frozenset(
    [
        (6, 193),
        (193, 55),
        (55, 107),
        (107, 66),
        (66, 105),
        (105, 63),
        (63, 70),
        (70, 124),
        (124, 226),
        (226, 110),
        (110, 24),
        (24, 23),
        (23, 22),
        (22, 26),
        (26, 233),
        (233, 128),
        (128, 114),
        (114, 188),
        (188, 122),
        (122, 196),
        (196, 3),
        (3, 236),
        (236, 198),  # right eye region
        (198, 209),
        (209, 49),
        (49, 129),
        (129, 98),
        (98, 167),
        (167, 37),
        (37, 39),
        (39, 40),
        (40, 186),
        (186, 57),
        (57, 43),
        (43, 106),
        (106, 182),
        (182, 83),
        (83, 18),
        (18, 313),
        (313, 406),
        (406, 335),
        (335, 273),
        (273, 287),
        (287, 410),
        (410, 270),
        (270, 269),
        (269, 267),
        (267, 393),
        (393, 327),
        (327, 358),
        (358, 279),
        (279, 429),
        (429, 420),
        (420, 456),
        (456, 248),
        (248, 419),
        (419, 351),
        (351, 412),
        (412, 343),
        (343, 357),
        (357, 453),
        (453, 256),  # nose region
        (256, 252),
        (252, 253),
        (253, 254),
        (254, 339),
        (339, 446),
        (446, 353),
        (353, 300),
        (300, 293),
        (293, 334),
        (334, 296),
        (296, 336),
        (336, 285),
        (285, 417),
        (417, 6),
    ]  # left eye region
)


# face contour
FACEMESH_FACE_OVAL = frozenset(
    [
        (10, 338),
        (338, 297),
        (297, 332),
        (332, 284),
        (284, 251),
        (251, 389),
        (389, 356),
        (356, 454),
        (454, 411),
        (411, 434),
        (434, 430),
        (430, 431),
        (431, 262),
        (262, 428),
        (428, 199),
        (199, 208),
        (208, 32),
        (32, 211),
        (211, 210),
        (210, 214),
        (214, 187),
        (187, 234),
        (234, 127),
        (127, 162),
        (162, 21),
        (21, 54),
        (54, 103),
        (103, 67),
        (67, 109),
        (109, 10),
    ]
)


def contour_fill_landmark(mesh, height, width):
    mask_image = np.zeros([height, width, 3], np.uint8)
    mpDraw = mp.solutions.drawing_utils
    drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    facial_landmarks = mesh.multi_face_landmarks[0]

    # eyes and nose contour
    mpDraw.draw_landmarks(
        image=mask_image,
        landmark_list=facial_landmarks,
        connections=FACEMESH_EYES_AND_NOSE,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_spec,
    )
    gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)

    _, im_th = cv2.threshold(gray_mask, 200, 255, cv2.THRESH_BINARY)
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_floodfill = im_th.copy()
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    return im_floodfill  # cv2.bitwise_and(im_floodfill, im_floodfill2)


def contour_fill_face(mesh, height, width):
    zero_img1 = np.zeros([height, width, 3], np.uint8)
    mpDraw = mp.solutions.drawing_utils
    drawing_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
    facial_landmarks = mesh.multi_face_landmarks[0]

    # face_oval
    mpDraw.draw_landmarks(
        image=zero_img1,
        landmark_list=facial_landmarks,
        connections=FACEMESH_FACE_OVAL,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_spec,
    )
    image = cv2.cvtColor(zero_img1, cv2.COLOR_RGB2GRAY)

    _, im_th = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im_floodfill_inv  # cv2.bitwise_and(im_floodfill, im_floodfill2),


def fill(mesh, height, width):
    face = contour_fill_face(mesh, height, width)
    landmark = contour_fill_landmark(mesh, height, width)
    return cv2.bitwise_and(landmark, face)


# FOREHEAD
def crop_forehead(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[103]
    r_t_x = int(rpt.x * width)

    lpt = facial_landmarks.landmark[332]
    l_t_x = int(lpt.x * width)

    # wider height
    wpt = facial_landmarks.landmark[10]
    w_t_y = int(wpt.y * height) - int(0.02 * height)

    # height
    up_pt = facial_landmarks.landmark[151]
    up_y = int(up_pt.y * height)

    down_pt = facial_landmarks.landmark[9]
    down_y = int(down_pt.y * height)

    half_y = int((down_y + up_y) / 2)

    #
    crop = image[w_t_y:half_y, r_t_x:l_t_x]

    # rgbcrop = rgbimage[w_t_y:half_y, r_t_x:l_t_x]

    # # Calculate RGB mean
    # avg_rgb = np.mean(rgbcrop, axis=(0,1))

    # print ("피부톤: ", avg_rgb)

    return crop, [w_t_y, half_y, r_t_x, l_t_x]


def crop_upnose(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[198]
    r_t_x = int(rpt.x * width)

    top_pt = facial_landmarks.landmark[188]
    r_t_y = int(top_pt.y * height)

    lpt = facial_landmarks.landmark[420]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[198]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# RIGHT upnose
# def crop_r_upnose(image, facial_landmarks, height, width):
#     # width
#     rpt = facial_landmarks.landmark[114]
#     r_t_x = int(rpt.x * width)

#     top_pt = facial_landmarks.landmark[188]
#     r_t_y = int(top_pt.y * height)

#     lpt = facial_landmarks.landmark[125]
#     l_t_x = int(lpt.x * width)

#     # height
#     down_pt = facial_landmarks.landmark[198]
#     down_y = int(down_pt.y * height)

#     crop = image[r_t_y:down_y, r_t_x:l_t_x]
#     return crop, [r_t_y, down_y, r_t_x, l_t_x]

# LEFT upnose
# def crop_l_upnose(image, facial_landmarks, height, width):
#     # width
#     rpt = facial_landmarks.landmark[354]
#     r_t_x = int(rpt.x * width)

#     top_pt = facial_landmarks.landmark[412]
#     r_t_y = int(top_pt.y * height)

#     lpt = facial_landmarks.landmark[343]
#     l_t_x = int(lpt.x * width)

#     # height
#     down_pt = facial_landmarks.landmark[420]
#     down_y = int(down_pt.y * height)

#     crop = image[r_t_y:down_y, r_t_x:l_t_x]
#     return crop, [r_t_y, down_y, r_t_x, l_t_x]


# GLABELLA
def crop_glabella(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[68]
    r_t_y = int(rpt.y * height)

    lpt = facial_landmarks.landmark[233]
    down_y = int(lpt.y * height)

    # height
    down_pt = facial_landmarks.landmark[245]
    l_t_x = int(down_pt.x * width)
    down_pt = facial_landmarks.landmark[465]
    r_t_x = int(down_pt.x * width)
    crop = image[r_t_y:down_y, l_t_x:r_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# RIGHT EYE SIDE
def crop_right_eye_side(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[139]
    r_t_x = int(rpt.x * width)
    r_t_y = int(rpt.y * height)

    lpt = facial_landmarks.landmark[31]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[111]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# RIGHT EYE UNDER
def crop_right_eye_under(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[35]
    r_t_x = int(rpt.x * width)
    # r_t_y = int(rpt.y * height)

    top_pt = facial_landmarks.landmark[23]
    r_t_y = int(top_pt.y * height)

    lpt = facial_landmarks.landmark[217]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[119]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# LEFT EYE SIDE
def crop_left_eye_side(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[448]
    r_t_x = int(rpt.x * width)

    lpt = facial_landmarks.landmark[368]
    l_t_x = int(lpt.x * width)
    l_t_y = int(lpt.y * height)

    # height
    down_pt = facial_landmarks.landmark[340]
    down_y = int(down_pt.y * height)

    crop = image[l_t_y:down_y, r_t_x:l_t_x]
    return crop, [l_t_y, down_y, r_t_x, l_t_x]


# LEFT EYE UNDER
def crop_left_eye_under(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[437]
    r_t_x = int(rpt.x * width)

    lpt = facial_landmarks.landmark[340]
    l_t_x = int(lpt.x * width)

    top_pt = facial_landmarks.landmark[253]
    l_t_y = int(top_pt.y * height)

    # height
    down_pt = facial_landmarks.landmark[437]
    down_y = int(down_pt.y * height)

    crop = image[l_t_y:down_y, r_t_x:l_t_x]
    return crop, [l_t_y, down_y, r_t_x, l_t_x]


# RIGHT CHEEK
def crop_right_cheek(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[50]
    r_t_x = int(rpt.x * width)

    lpt = facial_landmarks.landmark[64]
    l_t_x = int(lpt.x * width)
    down_y = int(lpt.y * height)

    # height
    down_pt = facial_landmarks.landmark[126]
    r_t_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# LEFT CHEEK
def crop_left_cheek(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[294]
    r_t_x = int(rpt.x * width)
    down_y = int(rpt.y * height)

    lpt = facial_landmarks.landmark[280]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[355]
    r_t_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# RIGHT philtrum
def crop_r_philtrum(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[184]
    r_t_x = int(rpt.x * width)

    top_pt = facial_landmarks.landmark[206]
    r_t_y = int(top_pt.y * height)

    lpt = facial_landmarks.landmark[167]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[37]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# LEFT philtrum
def crop_l_philtrum(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[393]
    r_t_x = int(rpt.x * width)

    top_pt = facial_landmarks.landmark[426]
    r_t_y = int(top_pt.y * height)

    lpt = facial_landmarks.landmark[408]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[267]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# RIGHT smile line
def crop_r_smile_line(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[207]
    r_t_x = int(rpt.x * width)

    lpt = facial_landmarks.landmark[64]
    l_t_x = int(lpt.x * width)
    top_pt = int(lpt.y * height)

    # height
    down_pt = facial_landmarks.landmark[92]
    down_y = int(down_pt.y * height)

    crop = image[top_pt:down_y, r_t_x:l_t_x]
    return crop, [top_pt, down_y, r_t_x, l_t_x]


# LEFT smile line
def crop_l_smile_line(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[294]
    r_t_x = int(rpt.x * width)
    r_t_y = int(rpt.y * height)

    lpt = facial_landmarks.landmark[427]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[322]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# RIGHT MOUTH
def crop_right_mouth(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[136]
    r_t_x = int(rpt.x * width)

    top_pt = facial_landmarks.landmark[214]
    r_t_y = int(top_pt.y * height)

    lpt = facial_landmarks.landmark[204]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[150]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# LEFT MOUTH
def crop_left_mouth(image, facial_landmarks, height, width):
    # width
    rpt = facial_landmarks.landmark[424]
    r_t_x = int(rpt.x * width)

    top_pt = facial_landmarks.landmark[434]
    r_t_y = int(top_pt.y * height)

    lpt = facial_landmarks.landmark[365]
    l_t_x = int(lpt.x * width)

    # height
    down_pt = facial_landmarks.landmark[379]
    down_y = int(down_pt.y * height)

    crop = image[r_t_y:down_y, r_t_x:l_t_x]
    return crop, [r_t_y, down_y, r_t_x, l_t_x]


# 1. 눈썹 명공(양쪽 눈썹 안쪽) 좌
# FuncPointTL(canvasCtx, landmarks[129], landmarks[358], landmarks[9], out_canvas);

#  상단 왼쪽


def point_tl(facial_landmarks):
    #  코 넓이 계산
    NoseWidth = facial_landmarks.landmark[358].x - facial_landmarks.landmark[129].x
    NoseCenter = NoseWidth / 2
    #  코 높이 계산
    NoseHeight = facial_landmarks.landmark[358].y - facial_landmarks.landmark[129].y
    NoseHCenter = NoseHeight / 2

    #  위치 저장
    point11 = []
    x = facial_landmarks.landmark[9].x - NoseCenter
    y = facial_landmarks.landmark[9].y - NoseHCenter
    point11.append(x)
    point11.append(y)
    return point11


# 1. 눈썹 명공(양쪽 눈썹 안쪽) 우
# FuncPointTR(canvasCtx, landmarks[129], landmarks[358], landmarks[9], out_canvas);

#  상단 오른쪽


def point_tr(facial_landmarks):
    #  코 넓이 계산
    NoseWidth = facial_landmarks.landmark[358].x - facial_landmarks.landmark[129].x
    NoseCenter = NoseWidth / 2
    #  코 높이 계산
    NoseHeight = facial_landmarks.landmark[358].y - facial_landmarks.landmark[129].y
    NoseHCenter = NoseHeight / 2

    #  위치 저장
    Point12 = [
        facial_landmarks.landmark[9].x - NoseCenter,
        facial_landmarks.landmark[9].y - NoseHCenter,
    ]
    print(Point12 + "*************************")
    return Point12


def crop_face(image):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mesh = face_mesh.process(rgb_image)
    facial_landmarks = mesh.multi_face_landmarks[0]
    height, width, _ = rgb_image.shape
    # width
    rpt = facial_landmarks.landmark[234]
    r_t_x = int(rpt.x * width) - int(0.1 * width)

    lpt = facial_landmarks.landmark[454]
    l_t_x = int(lpt.x * width) + int(0.1 * width)

    # wider height
    wpt = facial_landmarks.landmark[10]
    w_t_y = int(wpt.y * height) - int(0.1 * height)

    # height
    up_pt = facial_landmarks.landmark[152]
    up_y = int(up_pt.y * height) + int(0.1 * height)

    crop = image[w_t_y:up_y, r_t_x:l_t_x]
    crop = cv2.resize(crop, [512, 512])
    return crop


# hessian
def wrinkle_frangi(input):
    input = cv2.resize(input, [512, 512])
    out = cv2.GaussianBlur(input, (0, 0), 0.8)
    out = frangi(out, scale_range=(0, 2), scale_step=0.1, alpha=10, beta=10)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    _, out = cv2.threshold(out, 0.35, 255, cv2.THRESH_BINARY)
    out = cv2.erode(out, k)
    return out


def wrinkle_sato(input):
    input = cv2.resize(input, [1024, 1024])
    out = sato(input, (1, 10, 3))
    out = cv2.normalize(out, None, 0, 3, cv2.NORM_MINMAX)
    _, out = cv2.threshold(out, 0.1, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    out = cv2.erode(out, k)
    out = cv2.resize(out, [512, 512])
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    out = cv2.erode(out, k)
    return out


def wrinkle_detection(input, mesh):
    _, g, _ = cv2.split(input)
    sw = wrinkle_sato(g)
    fw = wrinkle_frangi(g)
    out = cv2.bitwise_and(sw.astype(np.uint8), fw.astype(np.uint8))
    l_region = fill(mesh, 512, 512)
    out = cv2.bitwise_and(out.astype(np.uint8), l_region.astype(np.uint8))
    return out


def visualization(face_img):
    cropped_img = crop_face(face_img)
    mesh = extract_mesh(cropped_img)
    wrinkle = wrinkle_detection(cropped_img, mesh)
    input = wrinkle
    funcs = [
        "crop_forehead",
        "crop_upnose",
        "crop_glabella",
        "crop_right_eye_side",
        "crop_left_eye_side",
        "crop_right_eye_under",
        "crop_left_eye_under",
        "crop_right_cheek",
        "crop_left_cheek",
        "crop_r_philtrum",
        "crop_l_philtrum",
        "crop_r_smile_line",
        "crop_l_smile_line",
        "crop_left_mouth",
        "crop_right_mouth",
    ]
    rgb_image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, [512, 512])
    copy_img = copy.deepcopy(rgb_image)
    for i in range(len(funcs)):
        _, rect = eval(
            "%s(rgb_image, mesh.multi_face_landmarks[0], 512, 512)" % funcs[i]
        )
        copy_img = cv2.rectangle(
            copy_img, (rect[2], rect[0]), (rect[3], rect[1]), (255, 0, 0), 3
        )
    plt.imshow(copy_img)
    plt.show()


def visualization_original(face_img):
    cropped_img = crop_face(face_img)
    mesh = extract_mesh(cropped_img)
    funcs = [
        "crop_forehead",
        "crop_upnose",
        "crop_glabella",
        "crop_right_eye_side",
        "crop_left_eye_side",
        "crop_right_eye_under",
        "crop_left_eye_under",
        "crop_right_cheek",
        "crop_left_cheek",
        "crop_r_philtrum",
        "crop_l_philtrum",
        "crop_r_smile_line",
        "crop_l_smile_line",
        "crop_left_mouth",
        "crop_right_mouth",
    ]
    rgb_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    # rgb_image = cv2.resize(rgb_image, [512, 512])
    copy_img = copy.deepcopy(rgb_image)
    for i in range(len(funcs)):
        _, rect = eval(
            "%s(copy_img, mesh.multi_face_landmarks[0], copy_img.shape[0], copy_img.shape[1])"
            % funcs[i]
        )
        copy_img = cv2.rectangle(
            copy_img, (rect[2], rect[0]), (rect[3], rect[1]), (255, 0, 0), 5
        )
    plt.imshow(copy_img)
    plt.show()


def extract_mesh(input):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    rgb_image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    # rgb_image = cv2.resize(rgb_image, [512, 512])
    mesh = face_mesh.process(rgb_image)
    return mesh


def compare_wrinkle(static, dynamic, s_mesh, d_mesh):
    funcs = [
        "crop_forehead",
        "crop_glabella",
        "crop_right_eye_side",
        "crop_left_eye_side",
        "crop_right_eye_under",
        "crop_left_eye_under",
        "crop_right_cheek",
        "crop_left_cheek",
        "crop_r_smile_line",
        "crop_l_smile_line",
        "crop_left_mouth",
        "crop_right_mouth",
    ]
    s_array = []
    d_array = []
    progress = []
    epsilon = 0.001
    for i in range(len(funcs)):
        s_cropped, _ = eval(
            "%s(static, s_mesh.multi_face_landmarks[0], 512, 512)" % funcs[i]
        )
        d_cropped, _ = eval(
            "%s(dynamic, d_mesh.multi_face_landmarks[0], 512, 512)" % funcs[i]
        )
        s_h, s_w = s_cropped.shape
        d_h, d_w = d_cropped.shape
        s_ratio = sum(sum(s_cropped)) / (s_h * s_w)
        d_ratio = sum(sum(d_cropped)) / (d_h * d_w)
        s_array.append(s_ratio)
        d_array.append(d_ratio)
        print(funcs[i])
        print(s_ratio)
        print(d_ratio)
        print(s_ratio / (d_ratio + epsilon) * 100)
        progress.append(s_ratio / (d_ratio + epsilon) * 100)
    return s_array, d_array, progress


# result < 0.2: 주름 없음
# 0.2 < result < 0.4: 약
# 0.4 < result < 0.6: 중
# 0.6 < result: 강


# 0~1: T-zone, 2~5: E-zone, 6~9: C-zone, 10~11: U-zone
def measure_wrinkle(face_img):
    try:
        cropped_img = crop_face(face_img)
        # cv2.imshow("croped img", cropped_img)
        # cv2.imshow("graymask",cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY) )
        mesh = extract_mesh(cropped_img)
        wrinkle = wrinkle_detection(cropped_img, mesh)

        funcs = [
            "crop_forehead",
            "crop_upnose",
            "crop_glabella",
            "crop_right_eye_side",
            "crop_left_eye_side",
            "crop_right_eye_under",
            "crop_left_eye_under",
            "crop_right_cheek",
            "crop_left_cheek",
            "crop_r_philtrum",
            "crop_l_philtrum",
            "crop_r_smile_line",
            "crop_l_smile_line",
            "crop_left_mouth",
            "crop_right_mouth",
        ]

        # sagfuncs = ["point_tl", "point_tr"]

        # 원본
        # ref = [12, 13, 25, 25, 25, 25, 20, 20, 20, 20, 20, 20, 10, 10]

        # ref = 각 부위별 주름 비율을 퍼센트로 변환 시 최대값
        ref = [12, 40, 12, 25, 25, 25, 25, 20, 20, 40, 40, 20, 20, 10, 10]

        progs = []
        data = ""
        measuredata = {}
        for i in range(len(funcs)):
            cropped, _ = eval(
                "%s(wrinkle, mesh.multi_face_landmarks[0], 512, 512)" % funcs[i]
            )
            h, w = cropped.shape
            s_ratio = (sum(sum(cropped / 255)) / (h * w)) * 100
            res = ""
            progs.append(s_ratio / ref[i])
            if progs[i] < 0.2:
                res = "주름 없음"
            elif progs[i] < 0.4:
                res = "약"
            elif progs[i] < 0.6:
                res = "중"
            else:
                res = "강"
            printdata = "%s: %s(%f)(%f)" % (funcs[i], res, progs[i], s_ratio)

            print(printdata)

            # data += '"%s": "%s(%f)",' %(funcs[i], res, progs[i])
            data += '"%s": "%f",' % (funcs[i], progs[i])
        measuredata = json.loads("{" + data.rstrip(",") + "}")

        print(measuredata)

        # for i in range(len(sagfuncs)):
        #     point, _ = eval("%s(mesh.multi_face_landmarks[0])" % sagfuncs[i])

        #     printdata = sagfuncs[i] + ": " + point

        #     print(printdata)

        return measuredata
    except:
        return "error"


def compar(img_list):
    funcs = [
        "crop_forehead",
        "crop_glabella",
        "crop_right_eye_side",
        "crop_left_eye_side",
        "crop_right_eye_under",
        "crop_left_eye_under",
        "crop_right_cheek",
        "crop_left_cheek",
        "crop_r_smile_line",
        "crop_l_smile_line",
        "crop_left_mouth",
        "crop_right_mouth",
    ]
    s_img = img_list[0]
    d_imgs = img_list[1:]
    s_score = measure_wrinkle(s_img)
    d_scores = []
    for d_img in d_imgs:
        d_scores.append(measure_wrinkle(d_img))
    max_d = np.max(d_scores, axis=0)
    progress = s_score / max_d * 100
    for i in range(len(funcs)):
        print("%s_진행도: %f" % (funcs[i], progress[i]))

    return progress


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image


def face_alignment(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load the face mesh detector
    mp_face_mesh = mp.solutions.face_mesh

    # Create a face mesh object
    face_mesh = mp_face_mesh.FaceMesh()

    # Detect the face landmarks in the image
    results = face_mesh.process(image)

    # Iterate over each face
    for face_landmarks in results.multi_face_landmarks:
        # Extract the coordinates of the left and right eyes
        left_eye_x = face_landmarks.landmark[159].x * image.shape[1]
        left_eye_y = face_landmarks.landmark[159].y * image.shape[0]
        right_eye_x = face_landmarks.landmark[386].x * image.shape[1]
        right_eye_y = face_landmarks.landmark[386].y * image.shape[0]

        # Calculate the angle between the eyes and the horizontal axis
        dx = right_eye_x - left_eye_x
        dy = right_eye_y - left_eye_y
        angle = cv2.fastAtan2(dy, dx)

        # Rotate the image around the center of the face
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(image, matrix, (w, h))

        # Convert the image back to BGR format
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
    return rotated_image


# ref = url_to_image('https://health.iot4health.co.kr:8300/images/photo/rainleex@iot4health.co.kr/20221213/A.jpg')
# _ = measure_wrinkle(ref)
# visualization(ref)


# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
# preds = fa.get_landmarks_from_directory('C:/Users/USER/Flask/img/gimg/person-people-woman-hair-old-portrait-profession-facial-expression-hairstyle-old-woman-senior-citizen-cuba-face-nose-strand-eye-head-organ-elderly-granny-wrinkled-1028683.jpg')

# facealignment
# ref = cv2.imread('C:/Users/USER/Flask/img/bright/Screenshot_20230327_095349.jpg')
# rotated_image = face_alignment(ref)
# _ = measure_wrinkle(rotated_image)
# visualization(rotated_image)


# none_facealignment
ref = cv2.imread(
    "C:/Users/USER/JM/wrinkle_RGB_flask/img/wrinkle_data/upnose/P15727932000075_15794873152813.jpg"
)
_ = measure_wrinkle(ref)
visualization_original(ref)

# vitadoc img
# ref = cv2.imread('//AMD/htdocs/ImgData/2017/0911/P15050880006971/P15050880006971_1505112500.jpg')
# _ = measure_wrinkle(ref)
# visualization(ref)
