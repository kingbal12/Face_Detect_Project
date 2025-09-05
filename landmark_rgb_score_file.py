import cv2
import mediapipe as mp
import numpy as np
import traceback

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = "C:/Users/USER/Flask/img/rgb/"
imgname = "IMG_0317"
photo_format = ".JPG"

type_1_max, type_1_min = 255, 232.7
type_2_max, type_2_min = 232.6, 218.7
type_3_max, type_3_min = 218.6, 191.3
type_4_max, type_4_min = 191.2, 130.7
type_5_max, type_5_min = 130.6, 46
type_6_max, type_6_min = 45.9, 6


# 시술 후 이미지 열기
def rgb_score():
    try:
        rgb_json = {}
        image = cv2.imread(path + imgname + photo_format)

        bgrimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(bgrimage)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # T-ZONE
                    def FOREHEAD():
                        top = int(image.shape[0] * face_landmarks.landmark[10].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[68].y)
                        right = int(image.shape[1] * face_landmarks.landmark[104].x)
                        left = int(image.shape[1] * face_landmarks.landmark[333].x)
                        return RGB_AVERAGE(top, bottom, right, left, "FOREHEAD")

                    def BETWEEN_THE_EYEBROWS():
                        top = int(image.shape[0] * face_landmarks.landmark[68].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[168].y)
                        right = int(image.shape[1] * face_landmarks.landmark[193].x)
                        left = int(image.shape[1] * face_landmarks.landmark[417].x)
                        return RGB_AVERAGE(
                            top, bottom, right, left, "BETWEEN_THE_EYEBROWS"
                        )

                    def NOSE():
                        top = int(image.shape[0] * face_landmarks.landmark[168].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[5].y)
                        right = int(image.shape[1] * face_landmarks.landmark[196].x)
                        left = int(image.shape[1] * face_landmarks.landmark[419].x)
                        return RGB_AVERAGE(top, bottom, right, left, "NOSE")

                    def NOSE_TIP():
                        top = int(image.shape[0] * face_landmarks.landmark[5].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[4].y)
                        right = int(image.shape[1] * face_landmarks.landmark[45].x)
                        left = int(image.shape[1] * face_landmarks.landmark[275].x)
                        return RGB_AVERAGE(top, bottom, right, left, "NOSE_TIP")

                    # E-ZONE
                    def RIGHT_EYE_RIGHT():
                        top = int(image.shape[0] * face_landmarks.landmark[156].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[116].y)
                        right = int(image.shape[1] * face_landmarks.landmark[143].x)
                        left = int(image.shape[1] * face_landmarks.landmark[68].x)
                        return RGB_AVERAGE(top, bottom, right, left, "RIGHT_EYE_RIGHT")

                    def LEFT_EYE_LEFT():
                        top = int(image.shape[0] * face_landmarks.landmark[156].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[116].y)
                        right = int(image.shape[1] * face_landmarks.landmark[298].x)
                        left = int(image.shape[1] * face_landmarks.landmark[372].x)
                        return RGB_AVERAGE(top, bottom, right, left, "LEFT_EYE_LEFT")

                    def UNDER_THE_EYE_RIGHT():
                        top = int(image.shape[0] * face_landmarks.landmark[231].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[119].y)
                        right = int(image.shape[1] * face_landmarks.landmark[228].x)
                        left = int(image.shape[1] * face_landmarks.landmark[121].x)
                        return RGB_AVERAGE(
                            top, bottom, right, left, "UNDER_THE_EYE_RIGHT"
                        )

                    def UNDER_THE_EYE_LEFT():
                        top = int(image.shape[0] * face_landmarks.landmark[231].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[119].y)
                        right = int(image.shape[1] * face_landmarks.landmark[350].x)
                        left = int(image.shape[1] * face_landmarks.landmark[448].x)
                        return RGB_AVERAGE(
                            top, bottom, right, left, "UNDER_THE_EYE_LEFT"
                        )

                    # C-ZONE
                    def RIGTH_CHEEK():
                        top = int(image.shape[0] * face_landmarks.landmark[119].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[192].y)
                        right = int(image.shape[1] * face_landmarks.landmark[192].x)
                        left = int(image.shape[1] * face_landmarks.landmark[119].x)
                        return RGB_AVERAGE(top, bottom, right, left, "RIGTH_CHEEK")

                    def LEFT_CHEEK():
                        top = int(image.shape[0] * face_landmarks.landmark[119].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[416].y)
                        right = int(image.shape[1] * face_landmarks.landmark[348].x)
                        left = int(image.shape[1] * face_landmarks.landmark[416].x)
                        return RGB_AVERAGE(top, bottom, right, left, "LEFT_CHEEK")

                    # U-ZONE
                    def RIGHT_CORNER_OF_MOUTH():
                        top = int(image.shape[0] * face_landmarks.landmark[192].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[135].y)
                        right = int(image.shape[1] * face_landmarks.landmark[118].x)
                        left = int(image.shape[1] * face_landmarks.landmark[202].x)
                        return RGB_AVERAGE(
                            top, bottom, right, left, "RIGHT_CORNER_OF_MOUTH"
                        )

                    def LEFT_CORNER_OF_MOUTH():
                        top = int(image.shape[0] * face_landmarks.landmark[416].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[135].y)
                        right = int(image.shape[1] * face_landmarks.landmark[422].x)
                        left = int(image.shape[1] * face_landmarks.landmark[347].x)
                        return RGB_AVERAGE(
                            top, bottom, right, left, "LEFT_CORNER_OF_MOUTH"
                        )

                    def RIGHT_CHIN():
                        top = int(image.shape[0] * face_landmarks.landmark[194].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[140].y)
                        right = int(image.shape[1] * face_landmarks.landmark[194].x)
                        left = int(image.shape[1] * face_landmarks.landmark[37].x)
                        return RGB_AVERAGE(top, bottom, right, left, "RIGHT_CHIN")

                    def LEFT_CHIN():
                        top = int(image.shape[0] * face_landmarks.landmark[194].y)
                        bottom = int(image.shape[0] * face_landmarks.landmark[140].y)
                        right = int(image.shape[1] * face_landmarks.landmark[267].x)
                        left = int(image.shape[1] * face_landmarks.landmark[418].x)
                        return RGB_AVERAGE(top, bottom, right, left, "LEFT_CHIN")

            def GET_SCORE(r, g, b):
                skintone = (r + g + b) / 3
                if skintone >= type_1_min and skintone <= type_1_max:
                    type = "type1"
                    score = 100 - (type_1_max - skintone) / (
                        (type_1_max - type_1_min) / 100
                    )
                elif skintone >= type_2_min and skintone <= type_2_max:
                    type = "type2"
                    score = 100 - (type_2_max - skintone) / (
                        (type_2_max - type_2_min) / 100
                    )
                elif skintone >= type_3_min and skintone <= type_3_max:
                    type = "type3"
                    score = 100 - (type_3_max - skintone) / (
                        (type_3_max - type_3_min) / 100
                    )
                elif skintone >= type_4_min and skintone <= type_4_max:
                    type = "type4"
                    score = 100 - (type_4_max - skintone) / (
                        (type_4_max - type_4_min) / 100
                    )
                elif skintone >= type_5_min and skintone <= type_5_max:
                    type = "type5"
                    score = 100 - (type_5_max - skintone) / (
                        (type_5_max - type_5_min) / 100
                    )
                elif skintone >= type_6_min and skintone <= type_6_max:
                    type = "type6"
                    score = 100 - (type_6_max - skintone) / (
                        (type_6_max - type_6_min) / 100
                    )
                else:
                    score = 0
                return [type, score]

            def RGB_AVERAGE(top, bottom, right, left, name):
                part_rgb_average_array = []
                cropped_img = image[top:bottom, right:left]

                rgb_values = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                # print("rgb_values : ", rgb_values)
                # print(rgb_values.ndim)
                # print(rgb_values.shape)

                # R, G, B 노이즈값 제거(털, 점, 머리카락)
                arr_list = rgb_values.tolist()

                for i in range(len(arr_list)):
                    for j in range(len(arr_list[i])):
                        if (
                            arr_list[i][j][0] <= 170
                            and arr_list[i][j][1] <= 170
                            and arr_list[i][j][2] <= 50
                        ):
                            arr_list[i][j] = [None, None, None]

                # new_arr_list = [
                #     sublst
                #     for sublst in arr_list
                #     if not all([ele is None for ele in sublst])
                # ]

                new_rgb_values = np.array(arr_list)

                # lst = rgb_values.tolist()

                # for i in lst[0]:
                #     if i[0] <= 150 and i[1] <= 75 and i[2] == 0:
                #         lst[0].remove(i)

                # new_rgb_values = np.array(lst)

                # print("new_rgb_values : ", new_rgb_values)
                # print(new_rgb_values.ndim)
                # print(new_rgb_values.shape)

                r, g, b = (
                    np.mean(rgb_values[:, :, 0]),
                    np.mean(rgb_values[:, :, 1]),
                    np.mean(rgb_values[:, :, 2]),
                )

                part_rgb_average_array.extend([r, g, b])

                rgb_json[name + "_TYPE"] = GET_SCORE(r, g, b)[0]
                rgb_json[name + "_SCORE"] = round(GET_SCORE(r, g, b)[1])
                rgb_json[name + "_RGB"] = part_rgb_average_array

                # draw_rectangle(image, left, top, right, bottom)
                return part_rgb_average_array

            def draw_rectangle(image, left, top, right, bottom):
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

            FOREHEAD()

            BETWEEN_THE_EYEBROWS()

            NOSE()

            NOSE_TIP()

            RIGHT_EYE_RIGHT()

            LEFT_EYE_LEFT()

            UNDER_THE_EYE_RIGHT()

            UNDER_THE_EYE_LEFT()

            RIGTH_CHEEK()

            LEFT_CHEEK()

            RIGHT_CORNER_OF_MOUTH()

            LEFT_CORNER_OF_MOUTH()

            RIGHT_CHIN()

            LEFT_CHIN()

            print(rgb_json)

            dst = cv2.resize(image, dsize=(640, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow("MediaPipe FaceMesh", dst)
            cv2.waitKey(0)
        return rgb_json
    except:
        traceback.print_exc()
        return "error"


rgb_score()
