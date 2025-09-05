import json
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import datetime


# 측정 포인트 설정
detect_points = [
    108,
    151,
    337,
    336,
    107,
    9,
    68,
    282,
    52,
    298,
    156,
    383,
    441,
    221,
    342,
    113,
    463,
    243,
    450,
    230,
    454,
    234,
    437,
    217,
    330,
    101,
    323,
    93,
    358,
    129,
    2,
    61,
    291,
    149,
    378,
]
imgnames = [
    "20230511_135347",
    "20230511_135348",
    "20230511_135349",
    "20230511_135350",
    "20230511_135351",
    "20230511_135352",
    "20230511_135353",
    "20230511_135354",
    "20230511_135355",
    "20230511_135356",
]
photo_format = ".jpg"


appended_data = []
for imgname in imgnames:
    # 이미지 열기
    image = cv2.imread("C:/Users/USER/Flask/img/pimg/" + imgname + photo_format)

    # 미디어파이프 초기화
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # 시술 전 사진 랜드마크 설정
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            raise ValueError("Face landmarks not found in the image.")

        landmark_points = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmark_points.append((landmark.x, landmark.y, landmark.z))

        # 사진에 기준점 찍기
        for detect_point in detect_points:
            x = landmark_points[detect_point][0]
            y = landmark_points[detect_point][1]

            shape = image.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])

            cv2.circle(
                image,
                (relative_x, relative_y),
                radius=1,
                color=(255, 255, 200),
                thickness=20,
            )

            cv2.putText(
                image,
                str(detect_point),
                (relative_x, relative_y),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )

        # 랜드마크 포인트 추출
        landmark_points = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmark_points.append((landmark.x, landmark.y, landmark.z))

        # 눈크기 (위쪽) 변화 계산
        # 랜드마크 2와 4의 길이 계산
        distance_24 = (
            (landmark_points[4][0] - landmark_points[2][0]) ** 2
            + (landmark_points[4][1] - landmark_points[2][1]) ** 2
            # + (landmark_points[1][2] - landmark_points[2][2]) ** 2
        ) ** 0.5

        # 구해진 랜드마크간 모든 길이의 비율 계산
        detect_points_ratio = {}
        for detect_point in detect_points:
            denominator = (
                (landmark_points[detect_point][0] - landmark_points[2][0]) ** 2
                + (landmark_points[detect_point][1] - landmark_points[2][1]) ** 2
            ) ** 0.5
            if denominator == 0:
                detect_points_ratio[detect_point] = 0
            else:
                ratios = denominator / distance_24
                detect_points_ratio[detect_point] = ratios

    # 기준점들의 비율 데이터 출력
    # plt.imshow(image)
    # plt.show()

    print("detect_points_ratio : ", detect_points_ratio)

    appended_data.append(detect_points_ratio)

# 파일 저장 날짜 설정
d = datetime.datetime.now()
filedate = d.strftime("%Y%m%d")
print("오늘 날짜: ", filedate)

# 여러 사진들의 기준점들의 비율 데이터 저장
print(appended_data)

file_path = "./" + filedate + "detect_points_data.json"

with open(file_path, "w") as outfile:
    json.dump(appended_data, outfile, indent=4)


# 기준점별 비율데이터 오차 평균 계산
err_range = []

for i in range(len(appended_data)):
    for key in appended_data[i]:
        if i == len(appended_data) - 1:
            break
        else:
            diff = np.abs(appended_data[i][key] - appended_data[i + 1][key])
            print(f"Key: {key}, Difference: {diff}")
            err_range.append({str(key): float(diff)})


err_range_average = {}
for i in err_range:
    for k, v in i.items():
        if k not in err_range_average:
            err_range_average[k] = [v]
        else:
            err_range_average[k].append(v)
for k, v in err_range_average.items():
    err_range_average[k] = sum(v) / len(v)
print(err_range_average)


# 기준점별 비율데이터 오차 평균 저장
file_path = "./" + filedate + "err_range_average.json"

with open(file_path, "w") as outfile:
    json.dump(err_range_average, outfile, indent=4)
