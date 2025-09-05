import cv2
import mediapipe as mp
import numpy as np
import datetime
import json


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

detect_point = 160

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

all_ratio = []
rgb_list = []

# 웹캠 초기화 부분
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    # connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )

                def FOREHEAD():
                    top = int(image.shape[0] * face_landmarks.landmark[10].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[68].y)
                    right = int(image.shape[1] * face_landmarks.landmark[104].x)
                    left = int(image.shape[1] * face_landmarks.landmark[333].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def BETWEEN_THE_EYEBROWS():
                    top = int(image.shape[0] * face_landmarks.landmark[68].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[168].y)
                    right = int(image.shape[1] * face_landmarks.landmark[193].x)
                    left = int(image.shape[1] * face_landmarks.landmark[417].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def NOSE():
                    top = int(image.shape[0] * face_landmarks.landmark[168].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[5].y)
                    right = int(image.shape[1] * face_landmarks.landmark[196].x)
                    left = int(image.shape[1] * face_landmarks.landmark[419].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                # E-ZONE
                def RIGHT_EYE_RIGHT():
                    top = int(image.shape[0] * face_landmarks.landmark[156].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[116].y)
                    right = int(image.shape[1] * face_landmarks.landmark[143].x)
                    left = int(image.shape[1] * face_landmarks.landmark[68].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def LEFT_EYE_LEFT():
                    top = int(image.shape[0] * face_landmarks.landmark[156].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[116].y)
                    right = int(image.shape[1] * face_landmarks.landmark[298].x)
                    left = int(image.shape[1] * face_landmarks.landmark[372].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def UNDER_THE_EYE_RIGHT():
                    top = int(image.shape[0] * face_landmarks.landmark[231].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[119].y)
                    right = int(image.shape[1] * face_landmarks.landmark[228].x)
                    left = int(image.shape[1] * face_landmarks.landmark[121].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def UNDER_THE_EYE_LEFT():
                    top = int(image.shape[0] * face_landmarks.landmark[231].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[119].y)
                    right = int(image.shape[1] * face_landmarks.landmark[350].x)
                    left = int(image.shape[1] * face_landmarks.landmark[448].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                # C-ZONE
                def RIGTH_CHEEK():
                    top = int(image.shape[0] * face_landmarks.landmark[119].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[192].y)
                    right = int(image.shape[1] * face_landmarks.landmark[192].x)
                    left = int(image.shape[1] * face_landmarks.landmark[119].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def LEFT_CHEEK():
                    top = int(image.shape[0] * face_landmarks.landmark[119].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[416].y)
                    right = int(image.shape[1] * face_landmarks.landmark[348].x)
                    left = int(image.shape[1] * face_landmarks.landmark[416].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                # U-ZONE
                def RIGHT_CORNER_OF_MOUTH():
                    top = int(image.shape[0] * face_landmarks.landmark[192].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[135].y)
                    right = int(image.shape[1] * face_landmarks.landmark[118].x)
                    left = int(image.shape[1] * face_landmarks.landmark[202].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def LEFT_CORNER_OF_MOUTH():
                    top = int(image.shape[0] * face_landmarks.landmark[416].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[135].y)
                    right = int(image.shape[1] * face_landmarks.landmark[422].x)
                    left = int(image.shape[1] * face_landmarks.landmark[347].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def RIGHT_CHIN():
                    top = int(image.shape[0] * face_landmarks.landmark[194].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[140].y)
                    right = int(image.shape[1] * face_landmarks.landmark[194].x)
                    left = int(image.shape[1] * face_landmarks.landmark[37].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def LEFT_CHIN():
                    top = int(image.shape[0] * face_landmarks.landmark[194].y)
                    bottom = int(image.shape[0] * face_landmarks.landmark[140].y)
                    right = int(image.shape[1] * face_landmarks.landmark[267].x)
                    left = int(image.shape[1] * face_landmarks.landmark[418].x)
                    return RGB_AVERAGE(top, bottom, right, left)

                def RGB_AVERAGE(top, bottom, right, left):
                    part_rgb_average_array = []
                    cropped_img = image[top:bottom, right:left]

                    rgb_values = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

                    # R, G, B 노이즈값 제거(털, 점, 머리카락)
                    # arr_list = rgb_values.tolist()

                    # for i in range(len(arr_list)):
                    #     for j in range(len(arr_list[i])):
                    #         if (
                    #             arr_list[i][j][0] <= 150
                    #             and arr_list[i][j][1] <= 75
                    #             and arr_list[i][j][2] == 0
                    #         ):
                    #             arr_list[i][j] = None

                    # arr_list = [[x for x in y if x is not None] for y in arr_list]

                    # new_rgb_values = np.array(arr_list)

                    r, g, b = (
                        np.mean(rgb_values[:, :, 0]),
                        np.mean(rgb_values[:, :, 1]),
                        np.mean(rgb_values[:, :, 2]),
                    )

                    part_rgb_average_array.extend([r, g, b])

                    # draw_rectangle(image, left, top, right, bottom)
                    return part_rgb_average_array

                def draw_rectangle(image, left, top, right, bottom):
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 3)

                rgb_json = {}

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
                    print("T_ZONE_RGB_AVERAGE : ", T_ZONE_RGB_AVERAGE)
                    rgb_json["T_ZONE_RGB_AVERAGE"] = T_ZONE_RGB_AVERAGE
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
                    print("E_ZONE_RGB_AVERAGE : ", E_ZONE_RGB_AVERAGE)
                    rgb_json["E_ZONE_RGB_AVERAGE"] = E_ZONE_RGB_AVERAGE
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
                    print("C_ZONE_RGB_AVERAGE : ", C_ZONE_RGB_AVERAGE)
                    rgb_json["C_ZONE_RGB_AVERAGE"] = C_ZONE_RGB_AVERAGE
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
                    print("U_ZONE_RGB_AVERAGE : ", U_ZONE_RGB_AVERAGE)
                    rgb_json["U_ZONE_RGB_AVERAGE"] = U_ZONE_RGB_AVERAGE
                    return U_ZONE_RGB_AVERAGE

                T_ZONE()
                E_ZONE()
                C_ZONE()
                U_ZONE()
                rgb_list.append(rgb_json)

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

            # 구해진 랜드마크간 길이의 비율 계산
            print(distance_2259 / distance_24)
            all_ratio.append(distance_2259 / distance_24)

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("웹캠", cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            print("rgb_list :", rgb_list)
            print("ratio : ", all_ratio)
            d = datetime.datetime.now()
            filedate = d.strftime("%Y%m%d%H")

            file_path = "./" + filedate + "rgb_list.json"

            with open(file_path, "w") as outfile:
                json.dump(rgb_list, outfile, indent=4)

            file_path2 = "./" + filedate + "all_ratio_list.json"

            with open(file_path2, "w") as outfile:
                json.dump(all_ratio, outfile, indent=4)
            break


cap.release()
