import cv2
import mediapipe as mp
import math

# 눈 동공의 좌표 인덱스
LEFT_EYE_INDEX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145]
RIGHT_EYE_INDEX = [362, 466, 388, 387, 386, 385, 384, 398, 362, 381, 380, 374, 467]

# MediaPipe 라이브러리의 FaceMesh 모듈을 초기화합니다.
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 얼굴에서 눈 동공 좌표를 추출하는 함수를 정의합니다.
def detect_eyes(image):
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:

        # 이미지에서 얼굴 랜드마크를 추출합니다.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 얼굴 랜드마크가 있다면, 눈 동공 좌표를 추출합니다.
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_eye_coords = []
            right_eye_coords = []
            for index in LEFT_EYE_INDEX:
                left_eye_coords.append(face_landmarks.landmark[index])
            for index in RIGHT_EYE_INDEX:
                right_eye_coords.append(face_landmarks.landmark[index])
            return left_eye_coords, right_eye_coords
        
# 두 눈 동공 사이의 거리를 계산하는 함수를 정의합니다.
def calculate_distance(left_eye_coords, right_eye_coords):
    # 각 눈 동공의 x, y 좌표를 추출합니다.
    left_eye_x = left_eye_coords[0].x * image_width
    left_eye_y = left_eye_coords[0].y * image_height
    right_eye_x = right_eye_coords[0].x * image_width
    right_eye_y = right_eye_coords[0].y * image_height
    
    # 두 눈 동공 사이의 거리를 계산합니다.
    distance = math.sqrt((left_eye_x - right_eye_x) ** 2 + (left_eye_y - right_eye_y) ** 2)
    return distance

# 이미지를 로드합니다.
image = cv2.imread("C:/Users/USER/Flask/img/P14516064002595_15787213082333_2M.png")
image_height, image_width, _ = image.shape

# 이미지에서 눈 동공 좌표를 추출합니다.
left_eye_coords, right_eye_coords = detect_eyes(image)

# 두 눈 동공 사이의 거리를 계산합니다.
distance = calculate_distance(left_eye_coords, right_eye_coords)

# 결과를 출력합니다.
print("두 눈 동공 사이의 거리: {:.2f} 픽셀".format(distance))
