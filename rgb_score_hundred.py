import cv2
import mediapipe as mp
import numpy as np
import traceback

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

path = "C:/Users/USER/Flask/img/rgb/"
imgname = "3before"
photo_format = ".JPG"

t_max, t_min = 241, 101
e_max, e_min = 226, 93
c_max, c_min = 216, 85
u_max, u_min = 200, 62


# 시술 후 이미지 열기
def rgb_score_p(image):
    try:
        rgb_json = {}
        # image = cv2.imread(path + imgname + photo_format)

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

            def RGB_AVERAGE(top, bottom, right, left, name):
                part_rgb_average_array = []
                cropped_img = image[top:bottom, right:left]

                rgb_values = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                # print("rgb_values : ", rgb_values)
                # print(rgb_values.ndim)
                # print(rgb_values.shape)

                # R, G, B 노이즈값 제거(털, 점, 머리카락)
                # arr_list = rgb_values.tolist()

                # for i in range(len(arr_list)):
                #     for j in range(len(arr_list[i])):
                #         if (
                #             arr_list[i][j][0] <= 170
                #             and arr_list[i][j][1] <= 170
                #             and arr_list[i][j][2] <= 50
                #         ):
                #             arr_list[i][j] = [None, None, None]

                r, g, b = (
                    np.mean(rgb_values[:, :, 0]),
                    np.mean(rgb_values[:, :, 1]),
                    np.mean(rgb_values[:, :, 2]),
                )

                part_rgb_average_array.extend([r, g, b])

                # draw_rectangle(image, left, top, right, bottom)
                return part_rgb_average_array

            def T_ZONE():
                T_ZONE_RGB_LIST = []
                T_ZONE_RGB_AVERAGE = []
                T_ZONE_RGB_LIST.append(FOREHEAD())
                T_ZONE_RGB_LIST.append(BETWEEN_THE_EYEBROWS())
                T_ZONE_RGB_LIST.append(NOSE())

                T_ZONE_RGB_ARRAY = np.array(T_ZONE_RGB_LIST)

                tr, tg, tb = (
                    np.mean(T_ZONE_RGB_ARRAY[:, 0]),
                    np.mean(T_ZONE_RGB_ARRAY[:, 1]),
                    np.mean(T_ZONE_RGB_ARRAY[:, 2]),
                )

                T_ZONE_RGB_AVERAGE.extend([tr, tg, tb])

                tskintone = (
                    T_ZONE_RGB_AVERAGE[0]
                    + T_ZONE_RGB_AVERAGE[1]
                    + T_ZONE_RGB_AVERAGE[2]
                ) / 3

                tscore = 0

                if tskintone < t_min:
                    tscore = 50
                elif tskintone > t_max:
                    tscore = 100
                else:
                    tscore = 50 + (tskintone - 101) * (50 / (t_max - t_min))

                rgb_json["T_ZONE_RGB_TONE"] = tskintone
                rgb_json["T_ZONE_RGB_SCORE"] = tscore

                return T_ZONE_RGB_AVERAGE

            def E_ZONE():
                E_ZONE_RGB_LIST = []
                E_ZONE_RGB_AVERAGE = []
                E_ZONE_RGB_LIST.append(RIGHT_EYE_RIGHT())
                E_ZONE_RGB_LIST.append(LEFT_EYE_LEFT())
                E_ZONE_RGB_LIST.append(UNDER_THE_EYE_RIGHT())
                E_ZONE_RGB_LIST.append(UNDER_THE_EYE_LEFT())

                E_ZONE_RGB_ARRAY = np.array(E_ZONE_RGB_LIST)

                er, eg, eb = (
                    np.mean(E_ZONE_RGB_ARRAY[:, 0]),
                    np.mean(E_ZONE_RGB_ARRAY[:, 1]),
                    np.mean(E_ZONE_RGB_ARRAY[:, 2]),
                )

                E_ZONE_RGB_AVERAGE.extend([er, eg, eb])

                eskintone = (
                    E_ZONE_RGB_AVERAGE[0]
                    + E_ZONE_RGB_AVERAGE[1]
                    + E_ZONE_RGB_AVERAGE[2]
                ) / 3

                escore = 0

                if eskintone < e_min:
                    escore = 50
                elif eskintone > e_max:
                    escore = 100
                else:
                    escore = 50 + (eskintone - 101) * (50 / (e_max - e_min))

                rgb_json["E_ZONE_RGB_TONE"] = eskintone
                rgb_json["E_ZONE_RGB_SCORE"] = escore

                return E_ZONE_RGB_AVERAGE

            def C_ZONE():
                C_ZONE_RGB_LIST = []
                C_ZONE_RGB_AVERAGE = []
                C_ZONE_RGB_LIST.append(RIGTH_CHEEK())
                C_ZONE_RGB_LIST.append(LEFT_CHEEK())

                C_ZONE_RGB_ARRAY = np.array(C_ZONE_RGB_LIST)

                cr, cg, cb = (
                    np.mean(C_ZONE_RGB_ARRAY[:, 0]),
                    np.mean(C_ZONE_RGB_ARRAY[:, 1]),
                    np.mean(C_ZONE_RGB_ARRAY[:, 2]),
                )

                C_ZONE_RGB_AVERAGE.extend([cr, cg, cb])

                cskintone = (
                    C_ZONE_RGB_AVERAGE[0]
                    + C_ZONE_RGB_AVERAGE[1]
                    + C_ZONE_RGB_AVERAGE[2]
                ) / 3

                cscore = 0

                if cskintone < c_min:
                    cscore = 50
                elif cskintone > c_max:
                    cscore = 100
                else:
                    cscore = 50 + (cskintone - 101) * (50 / (c_max - c_min))

                rgb_json["C_ZONE_RGB_TONE"] = cskintone
                rgb_json["C_ZONE_RGB_SCORE"] = cscore

                return C_ZONE_RGB_AVERAGE

            def U_ZONE():
                U_ZONE_RGB_LIST = []
                U_ZONE_RGB_AVERAGE = []
                U_ZONE_RGB_LIST.append(RIGHT_CORNER_OF_MOUTH())
                U_ZONE_RGB_LIST.append(LEFT_CORNER_OF_MOUTH())
                U_ZONE_RGB_LIST.append(RIGHT_CHIN())
                U_ZONE_RGB_LIST.append(LEFT_CHIN())

                U_ZONE_RGB_ARRAY = np.array(U_ZONE_RGB_LIST)

                ur, ug, ub = (
                    np.mean(U_ZONE_RGB_ARRAY[:, 0]),
                    np.mean(U_ZONE_RGB_ARRAY[:, 1]),
                    np.mean(U_ZONE_RGB_ARRAY[:, 2]),
                )

                U_ZONE_RGB_AVERAGE.extend([ur, ug, ub])

                uskintone = (
                    U_ZONE_RGB_AVERAGE[0]
                    + U_ZONE_RGB_AVERAGE[1]
                    + U_ZONE_RGB_AVERAGE[2]
                ) / 3

                uscore = 0

                if uskintone < u_min:
                    uscore = 50
                elif uskintone > u_max:
                    uscore = 100
                else:
                    uscore = 50 + (uskintone - 101) * (50 / (u_max - u_min))

                rgb_json["U_ZONE_RGB_TONE"] = uskintone
                rgb_json["U_ZONE_RGB_SCORE"] = uscore

                return U_ZONE_RGB_AVERAGE

            T_ZONE()
            E_ZONE()
            C_ZONE()
            U_ZONE()

            print(rgb_json)

            return rgb_json
    except:
        traceback.print_exc()
        return "error"
