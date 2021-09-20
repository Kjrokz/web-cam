from argparse import ArgumentParser
import numpy as np
from scipy.linalg.matfuncs import coshm
import cv2
import onnxruntime
import sys
from pathlib import Path
import constants

# local imports
from utils import FaceDetector, draw_axis

face_d = FaceDetector()

sess = onnxruntime.InferenceSession("models/fsanet-1x1-iter-688590.onnx")
sess2 = onnxruntime.InferenceSession("models/fsanet-var-iter-688590.onnx")
##print("Processing frames, press q to exit application...")


def direction(frame):

    ##cap = cv2.VideoCapture(1)
    ##cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # 3cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ##ret, frame = cap.read()

    # get face bounding boxes from frame

    cv2.putText(
        frame,
        "Cabeceos : {}".format(constants.HEAD),
        (20, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (224, 222, 96),
        2,
    )

    face_bb = face_d.get(frame)
    for (x1, y1, x2, y2) in face_bb:
        face_roi = frame[y1 : y2 + 1, x1 : x2 + 1]

        # preprocess headpose model input
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.transpose((2, 0, 1))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = (face_roi - 127.5) / 128
        face_roi = face_roi.astype(np.float32)

        # get headpose
        res1 = sess.run(["output"], {"input": face_roi})[0]
        res2 = sess2.run(["output"], {"input": face_roi})[0]

        # print("res1 : {}".format(res1))
        # print("res2 : {}".format(res2))

        yaw, pitch, roll = np.mean(np.vstack((res1, res2)), axis=0)

        #print("yaw : {}".format(yaw))
        #print("pitch : {}".format(pitch))
        #print("roll : {}".format(roll))

        draw_axis(
            frame,
            yaw,
            pitch,
            roll,
            tdx=(x2 - x1) // 2 + x1,
            tdy=(y2 - y1) // 2 + y1,
            size=50,
        )

        if pitch > -20 and pitch < 20 and yaw > -30 and yaw < 30:
            cv2.putText(
                frame,
                "centro",
                (190, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (224, 222, 96),
                2,
            )
            if constants.MAS >= 4 and constants.MAS <= 10:
                constants.HEAD = constants.HEAD + 1
                constants.MAS = 0
            constants.MAS = 0
        if pitch < -20:
            cv2.putText(
                frame,
                "abajo",
                (190, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (224, 222, 96),
                2,
            )
            constants.MAS = constants.MAS + 1
            #print("mas : {}".format(constants.MAS))

        if pitch > 20:
            cv2.putText(
                frame,
                "arriba",
                (190, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (224, 222, 96),
                2,
            )
            constants.MAS = 0
        if yaw > 30:
            cv2.putText(
                frame,
                "derecha",
                (190, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (224, 222, 96),
                2,
            )
            constants.MAS = 0
        if yaw < -30:
            cv2.putText(
                frame,
                "izquierda",
                (190, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (224, 222, 96),
                2,
            )
            constants.MAS = 0

        # draw face bb

        ##cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        ##cv2.imshow("Frame", frame)

        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        # break
