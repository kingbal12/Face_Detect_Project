import cv2
import mediapipe as mp

# 이미지 로드
orimage = cv2.imread('C:/Users/USER/JM/wrinkle_RGB_flask/img/gimg/21430553_1838586886159070_977274737029107045_n.jpg')

# 이미지를 RGB포맷으로 변경
image = cv2.cvtColor(orimage, cv2.COLOR_BGR2RGB)

# 미디어파이프의 페이스메쉬 로드
mp_face_mesh = mp.solutions.face_mesh

# 페이스메쉬 객체 만들기
face_mesh = mp_face_mesh.FaceMesh()

# 이미지상 랜드마크 감지
results = face_mesh.process(image)

# 반복문을 통해 랜드마크 생성
for face_landmarks in results.multi_face_landmarks:
    # 좌측, 우측 눈 좌표 추출
    left_eye_x = face_landmarks.landmark[159].x * image.shape[1]
    left_eye_y = face_landmarks.landmark[159].y * image.shape[0]
    right_eye_x = face_landmarks.landmark[386].x * image.shape[1]
    right_eye_y = face_landmarks.landmark[386].y * image.shape[0]

    # 눈과 수평축 사이의 각도 계산
    dx = right_eye_x - left_eye_x
    dy = right_eye_y - left_eye_y
    angle = cv2.fastAtan2(dy, dx)

    # 얼굴 중앙을 중심으로 이미지 회전
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h))

    # 이미지를 BGR 포맷으로 변경
    rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2GRAY)

    # 오리지널 이미지와 회전된 이미지를 출력
    cv2.imshow('Original Image', orimage)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    break

# 이미지 출력창을 종료시 프로그램 종료
cv2.destroyAllWindows()
