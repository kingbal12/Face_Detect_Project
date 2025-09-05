import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import traceback

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = "C:/Users/USER/Flask/img/rgb/"
imgname = "IMG_0088"
photo_format = ".JPG"


# 시술 후 이미지 열기
def find_rgb(path):
    # try:
    image = cv2.imread(path + imgname + photo_format)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print("No face detected.")
            return

        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        def NOSE_TIP():
            top = int(image.shape[0] * landmarks[5][1])
            bottom = int(image.shape[0] * landmarks[4][1])
            right = int(image.shape[1] * landmarks[45][0])
            left = int(image.shape[1] * landmarks[275][0])
            return BRIGHTNESS_AVERAGE(top, bottom, right, left)

        def BRIGHTNESS_AVERAGE(top, bottom, right, left):
            cropped_img = image[top:bottom, right:left]

            gray_image = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray_image.mean()

            # draw_rectangle(image, left, top, right, bottom)
            print(mean_brightness)
            return mean_brightness

        def draw_rectangle(image, left, top, right, bottom):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

        NOSE_TIP()

        dst = cv2.resize(image, dsize=(640, 800), interpolation=cv2.INTER_AREA)
        cv2.imshow("MediaPipe FaceMesh", dst)
        cv2.waitKey(0)

        change_brightness = 15
        im = Image.open(path + imgname + photo_format)

        pix = np.array(im)

        for i in range(pix.shape[0]):
            for j in range(pix.shape[1]):
                if (
                    pix[i][j][0] + change_brightness < 255
                    and pix[i][j][1] + change_brightness < 255
                    and pix[i][j][2] + change_brightness < 255
                ).any():
                    pix[i][j] += change_brightness  # 이 부분에서 원하는 밝기값만큼 올립니다.
                else:
                    pix[i][j][0], pix[i][j][1], pix[i][j][2] = 255, 255, 255

        new_im = Image.fromarray(pix)

        new_im.save("new_image.jpg")


find_rgb(path)
