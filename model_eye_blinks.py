from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib
import cv2
from constants import EYE_AR_CONSEC_FRAMES, EYE_AR_THRESH
import constants


# Inicializar el contador

FACIAL_LANDMARKS_68_IDXS = OrderedDict(
    [
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17)),
    ]
)


def eye_aspect_ratio(eye):
    # Calcular distancia, vertical
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Calcular distancia, horizontal
    C = dist.euclidean(eye[0], eye[3])
    # valor del oído
    ear = (A + B) / (2.0 * C)
    return ear


ap = argparse.ArgumentParser()
ap.add_argument(
    "-p",
    "--shape-predictor",
    default="models/shape_predictor_68_face_landmarks.dat",
    help="path to facial landmark predictor",
)
ap.add_argument(
    "-v", "--video", type=str, default="test.mp4", help="path to input video file"
)
args = vars(ap.parse_args())


# Herramientas de detección y posicionamiento
# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Tome dos áreas de los ojos por separado
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]


def shape_to_np(shape, dtype="int"):
    # Crear 68 * 2
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    # Recorriendo cada punto clave
    # Obtener coordenadas
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def model_eye_blinks(frame):

    # 3print(constants.TOTAL)
    ##print(constants.COUTER)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detectar caras
    rects = detector(gray, 0)

    # Atraviesa cada cara detectada
    for rect in rects:
        # Obtener coordenadas
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # Calcular los valores del oído por separado
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # Calcular un promedio
        ear = (leftEAR + rightEAR) / 2.0

        # Dibujar el área de los ojos
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        ##print("left eye : {}".format(leftEye))
        ##print("right eye : {}".format(rightEye))
        cv2.drawContours(frame, [leftEyeHull], -1, (183, 96, 224), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (183, 96, 224), 1)

        # Compruebe si se alcanza el umbral
        if ear < EYE_AR_THRESH:
            constants.COUTER += 1
            ##print("couter : {}".format(constants.COUTER))

            if constants.COUTER > 6:
                cv2.putText(
                    frame,
                    "Ojos cerrados",
                    (300, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (224, 222, 96),
                    2,
                )
        else:

            # Si los ojos están cerrados durante varios cuadros consecutivos, se cuenta el total
            if constants.COUTER >= EYE_AR_CONSEC_FRAMES:
                constants.TOTAL += 1
                ##print("total : {}".format(constants.TOTAL))

            # Restablecer
            constants.COUTER = 0

        # Mostrar
        cv2.putText(
            frame,
            "Parpadeos: {}".format(constants.TOTAL),
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (34, 153, 84),
            2,
        )
        cv2.putText(
            frame,
            "Umbral: {:.2f}".format(ear),
            (300, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (34, 153, 84),
            2,
        )
