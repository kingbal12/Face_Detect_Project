import json
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 측정 포인트 설정
detect_point = 160
before = "KakaoTalk_20230503_101229305"
after = "KakaoTalk_20230503_101229305_01"
photo_format = ".jpg"


# 이미지 열기
image = cv2.imread("C:/Users/USER/Flask/img/pimg/" + before + photo_format)


# 미디어파이프 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 크롭 함수
# def crop_face(image):
#     mp_face_mesh = mp.solutions.face_mesh
#     face_mesh = mp_face_mesh.FaceMesh()
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     mesh = face_mesh.process(rgb_image)
#     facial_landmarks = mesh.multi_face_landmarks[0]
#     height, width, _ = rgb_image.shape
#     # width
#     rpt = facial_landmarks.landmark[234]
#     r_t_x = int(rpt.x * width) - int(0.1 * width)

#     lpt = facial_landmarks.landmark[454]
#     l_t_x = int(lpt.x * width) + int(0.1 * width)

#     # wider height
#     wpt = facial_landmarks.landmark[10]
#     w_t_y = int(wpt.y * height) - int(0.1 * height)

#     # height
#     up_pt = facial_landmarks.landmark[152]
#     up_y = int(up_pt.y * height) + int(0.1 * height)

#     crop = image[w_t_y:up_y, r_t_x:l_t_x]
#     crop = cv2.resize(crop, [512, 512])
#     return crop


# 시술 전 사진 랜드마크 설정
with mp_face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
) as face_mesh:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        raise ValueError("Face landmarks not found in the image.")

    facial_landmarks = results.multi_face_landmarks[0]
    height, width, _ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).shape
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

    # 랜드마크 점 찍기
    for face in results.multi_face_landmarks:
        for landmark in face.landmark:
            x = landmark.x
            y = landmark.y

            shape = image.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])

            cv2.circle(
                image,
                (relative_x, relative_y),
                radius=1,
                color=(225, 0, 100),
                thickness=10,
            )

    # 랜드마크 포인트 추출
    landmark_points = []
    for landmark in results.multi_face_landmarks[0].landmark:
        landmark_points.append((landmark.x, landmark.y, landmark.z))

    # 기준 랜드마크 길이 계산
    distance_24 = (
        (landmark_points[4][0] - landmark_points[2][0]) ** 2
        + (landmark_points[4][1] - landmark_points[2][1]) ** 2
        # + (landmark_points[1][2] - landmark_points[2][2]) ** 2
    ) ** 0.5

    # 측정 랜드마크 길이 계산
    distance_2259 = (
        (landmark_points[detect_point][0] - landmark_points[2][0]) ** 2
        + (landmark_points[detect_point][1] - landmark_points[2][1]) ** 2
        # + (landmark_points[260][2] - landmark_points[2][2]) ** 2
    ) ** 0.5

    # 측정된 랜드마크간 거리를 선으로 표현
    shape = image.shape

    cv2.line(
        image,
        (int(landmark_points[2][0] * shape[1]), int(landmark_points[2][1] * shape[0])),
        (
            int(landmark_points[4][0] * shape[1]),
            int(landmark_points[4][1] * shape[0]),
        ),
        color=(225, 0, 100),
        thickness=5,
    )

    cv2.line(
        image,
        (int(landmark_points[2][0] * shape[1]), int(landmark_points[2][1] * shape[0])),
        (
            int(landmark_points[detect_point][0] * shape[1]),
            int(landmark_points[detect_point][1] * shape[0]),
        ),
        color=(225, 0, 100),
        thickness=5,
    )

    # 구해진 랜드마크간 길이의 비율 계산
    ratio = distance_2259 / distance_24

    # 구해진 랜드마크간 모든 길이의 비율 계산
    ratio_all = {}
    for i in range(len(landmark_points)):
        denominator = (
            (landmark_points[i][0] - landmark_points[2][0]) ** 2
            + (landmark_points[i][1] - landmark_points[2][1]) ** 2
        ) ** 0.5
        if denominator == 0:
            ratio_all[i] = 0
        else:
            ratios = denominator / distance_24
            ratio_all[i] = ratios

    json_ratio_all = json.dumps(ratio_all)


# 시술 전 비율
print("시술 전 : ", ratio)

plt.imshow(image)
plt.show()


# 시술 후 이미지 열기
image_after = cv2.imread("C:/Users/USER/Flask/img/pimg/" + after + photo_format)

# 시술 후 미디어파이프 초기화
mp_drawing_after = mp.solutions.drawing_utils
mp_face_mesh_after = mp.solutions.face_mesh

# 시술 후 사진 랜드마크 설정
with mp_face_mesh_after.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
) as face_mesh:
    results_after = face_mesh.process(cv2.cvtColor(image_after, cv2.COLOR_BGR2RGB))

    if not results_after.multi_face_landmarks:
        raise ValueError("Face landmarks not found in the image.")

    # 랜드마크 점 찍기
    for face_after in results_after.multi_face_landmarks:
        for landmark_after in face_after.landmark:
            x = landmark_after.x
            y = landmark_after.y

            shape = image_after.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])

            cv2.circle(
                image_after,
                (relative_x, relative_y),
                radius=1,
                color=(225, 0, 100),
                thickness=10,
            )

    # 랜드마크 포인트 추출
    landmark_points_after = []
    for landmark_after in results_after.multi_face_landmarks[0].landmark:
        landmark_points_after.append(
            (landmark_after.x, landmark_after.y, landmark_after.z)
        )

    # 기준 랜드마크 길이 계산
    distance_24_after = (
        (landmark_points_after[4][0] - landmark_points_after[2][0]) ** 2
        + (landmark_points_after[4][1] - landmark_points_after[2][1]) ** 2
        # + (landmark_points[4][2] - landmark_points[2][2]) ** 2
    ) ** 0.5

    # 측정 랜드마크 길이 계산
    distance_2259_after = (
        (landmark_points_after[detect_point][0] - landmark_points_after[2][0]) ** 2
        + (landmark_points_after[detect_point][1] - landmark_points_after[2][1]) ** 2
        # + (landmark_points[259][2] - landmark_points[2][2]) ** 2
    ) ** 0.5

    # 측정된 랜드마크간 거리를 선으로 표현
    shape_after = image_after.shape

    cv2.line(
        image_after,
        (
            int(landmark_points_after[2][0] * shape_after[1]),
            int(landmark_points_after[2][1] * shape_after[0]),
        ),
        (
            int(landmark_points_after[4][0] * shape_after[1]),
            int(landmark_points_after[4][1] * shape_after[0]),
        ),
        color=(225, 0, 100),
        thickness=5,
    )

    cv2.line(
        image_after,
        (
            int(landmark_points_after[2][0] * shape_after[1]),
            int(landmark_points_after[2][1] * shape_after[0]),
        ),
        (
            int(landmark_points_after[detect_point][0] * shape_after[1]),
            int(landmark_points_after[detect_point][1] * shape_after[0]),
        ),
        color=(225, 0, 100),
        thickness=5,
    )

    # 구해진 랜드마크간 길이의 비율 계산
    ratio_after = distance_2259_after / distance_24_after

    ratio_all_after = {}
    for i in range(len(landmark_points_after)):
        denominator_after = (
            (landmark_points_after[i][0] - landmark_points_after[2][0]) ** 2
            + (landmark_points_after[i][1] - landmark_points_after[2][1]) ** 2
        ) ** 0.5
        if denominator_after == 0:
            ratio_all_after[i] = 0
        else:
            ratios_after = denominator_after / distance_24_after
            ratio_all_after[i] = ratios_after

    json_ratio_all_after = json.dumps(ratio_all_after)


plt.imshow(image_after)
plt.show()


# 시술 후 비율
print("시술 후 : ", ratio_after)


diff = {}
for key in json.loads(json_ratio_all).keys():
    diff[key] = abs(
        json.loads(json_ratio_all)[key] - json.loads(json_ratio_all_after)[key]
    )

sorted_diff = sorted(diff.items(), key=lambda x: x[1], reverse=True)

json.dumps(sorted_diff)

# print("\n차이점 소팅\n ", sorted_diff)


# 소팅 데이터 저장
file_path = "./" + after + "_" + str(detect_point) + "_point_sorted_diff_data.json"

with open(file_path, "w") as outfile:
    json.dump(sorted_diff, outfile)


def landmark_distance(s_pointx, s_pointy, m_pointx, m_pointy):
    distance = ((m_pointx - s_pointx) ** 2 + (m_pointy - s_pointy) ** 2) ** 0.5

    return distance


def distance_rate_diff():
    dratediff = (
        landmark_distance() / landmark_distance()
        - landmark_distance() / landmark_distance()
    )

    return dratediff


