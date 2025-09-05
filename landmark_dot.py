import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

# 측정 포인트 설정
dotpoints = [
    # 334,
    # 105,
    # 108,
    # 151,
    # 337,
    # 336,
    # 107,
    # 9,
    # 68,
    # 282,
    # 52,
    # 298,
    # 156,
    # 383,
    # 441,
    # 221,
    # 342,
    # 113,
    # 463,
    # 243,
    # 450,
    # 230,
    # 454,
    # 234,
    # 437,
    # 217,
    # 330,
    # 101,
    # 323,
    # 93,
    # 358,
    # 129,
    # 2,
    # 61,
    # 291,
    # 149,
    # 378,
    # 9,
    129,
    358,
]
after = "img2"
photo_format = ".jpg"

# 시술 후 이미지 열기
image = cv2.imread("C:/Users/USER/Flask/img/" + after + photo_format)

# 시술 후 미디어파이프 초기화
mp_drawing_after = mp.solutions.drawing_utils
mp_face_mesh_after = mp.solutions.face_mesh

# 시술 후 사진 랜드마크 설정
with mp_face_mesh_after.FaceMesh(
    static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
) as face_mesh:
    results_after = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results_after.multi_face_landmarks:
        raise ValueError("Face landmarks not found in the image.")

    # 랜드마크 포인트 추출
    landmark_points_after = []
    for landmark_after in results_after.multi_face_landmarks[0].landmark:
        landmark_points_after.append(
            (landmark_after.x, landmark_after.y, landmark_after.z)
        )


# 원하는 포인트 점 찍기
for dotpoint in dotpoints:
    x = landmark_points_after[dotpoint][0]
    y = landmark_points_after[dotpoint][1]

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
        str(dotpoint),
        (relative_x, relative_y),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    # 눈썹 명공(양쪽 눈썹 안쪽) 좌
    def FuncPointTL(landmarkLeft, landmarkRight, landmarks):
        #  코 넓이 계산
        NoseWidth = landmarkRight[0] - landmarkLeft[0]
        NoseCenter = NoseWidth / 2
        #  코 높이 계산
        NoseHeight = landmarkRight[1] - landmarkLeft[1]
        NoseHCenter = NoseHeight / 2

        #  위치 저장
        Point11 = [landmarks[0] - NoseCenter, landmarks[1] - NoseHCenter]
        print(Point11)
        draw_dot(image, Point11[0], Point11[1])

    # 눈썹 명공(양쪽 눈썹 안쪽) 우
    def FuncPointTR(landmarkLeft, landmarkRight, landmarks):
        #  코 넓이 계산
        NoseWidth = landmarkRight[0] - landmarkLeft[0]
        NoseCenter = NoseWidth / 2
        #  코 높이 계산
        NoseHeight = landmarkRight[1] - landmarkLeft[1]
        NoseHCenter = NoseHeight / 2
        #  위치 저장
        Point12 = [landmarks[0] + NoseCenter, landmarks[1] - NoseHCenter]
        print(Point12)
        draw_dot(image, Point12[0], Point12[1])

    def draw_dot(image, x, y):
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])

        cv2.circle(
            image,
            (relative_x, relative_y),
            radius=1,
            color=(255, 255, 200),
            thickness=20,
        )


FuncPointTL(
    landmark_points_after[129], landmark_points_after[358], landmark_points_after[9]
)

FuncPointTR(
    landmark_points_after[129], landmark_points_after[358], landmark_points_after[9]
)


plt.imshow(image)
plt.show()
