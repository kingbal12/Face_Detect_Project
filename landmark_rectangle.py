import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

imgname = "img2"
photo_format = ".jpg"

# 시술 후 이미지 열기
image = cv2.imread("C:/Users/USER/Flask/img/" + imgname + photo_format)

bgrimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as face_mesh:
    results = face_mesh.process(bgrimage)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            def set_rectangle_zone():
                top = int(image.shape[0] * face_landmarks.landmark[23].y)
                bottom = int(image.shape[0] * face_landmarks.landmark[119].y)
                right = int(image.shape[1] * face_landmarks.landmark[35].x)
                left = int(image.shape[1] * face_landmarks.landmark[217].x)
                return RGB_AVERAGE(top, bottom, right, left)

            def set_rectangle_zone2():
                top = int(image.shape[0] * face_landmarks.landmark[253].y)
                bottom = int(image.shape[0] * face_landmarks.landmark[437].y)
                right = int(image.shape[1] * face_landmarks.landmark[437].x)
                left = int(image.shape[1] * face_landmarks.landmark[340].x)
                return RGB_AVERAGE(top, bottom, right, left)

    def RGB_AVERAGE(top, bottom, right, left):
        part_rgb_average_array = []
        cropped_img = image[top:bottom, right:left]

        rgb_values = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        r, g, b = (
            np.mean(rgb_values[:, :, 0]),
            np.mean(rgb_values[:, :, 1]),
            np.mean(rgb_values[:, :, 2]),
        )

        part_rgb_average_array.extend([r, g, b])

        draw_rectangle(image, left, top, right, bottom)
        return part_rgb_average_array

    def draw_rectangle(image, left, top, right, bottom):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

    set_rectangle_zone()
    set_rectangle_zone2()
    dst = cv2.resize(image, dsize=(640, 800), interpolation=cv2.INTER_AREA)
    cv2.imshow("MediaPipe FaceMesh", dst)
    cv2.waitKey(0)
    plt.imshow(image)
    plt.show()
