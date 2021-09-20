from pathlib import Path
import glob
import time
import numpy as np
import cv2
import numpy as np

root_path = Path(__file__).parent.parent


class FaceDetector:
    """
    This class is used for detecting face.
    """

    def __init__(self):

        """
        Constructor of class
        """

        config_path = root_path.joinpath("models/", "resnet10_ssd.prototxt")
        face_model_path = root_path.joinpath(
            "models/", "res10_300x300_ssd_iter_140000.caffemodel"
        )

        print(config_path)
        print(face_model_path)

        self.detector = cv2.dnn.readNetFromCaffe(
            str("models/resnet10_ssd.prototxt"),
            str("models/res10_300x300_ssd_iter_140000.caffemodel"),
        )

        # detector prediction threshold
        self.confidence = 0.7

    def get(self, img):
        """
        Given a image, detect faces and compute their bb

        """
        bb = self._detect_face_ResNet10_SSD(img)

        return bb

    def _detect_face_ResNet10_SSD(self, img):
        """
        Given a img, detect faces in it using resnet10_ssd detector

        """

        detector = self.detector
        (h, w) = img.shape[:2]
        # construct a blob from the image
        img_blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        detector.setInput(img_blob)
        detections = detector.forward()

        (start_x, start_y, end_x, end_y) = (0, 0, 0, 0)
        faces_bb = []
        if len(detections) > 0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            for i in range(0, detections.shape[2]):

                score = detections[0, 0, i, 2]

                # ensure that the detection greater than our threshold is
                # selected
                if score > self.confidence:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    box = box.astype("int")
                    (start_x, start_y, end_x, end_y) = box

                    # print("start x : {}".format(start_x))
                    # print("start y : {}".format(start_y))
                    # print("end x : {}".format(end_x))
                    # print("end y : {}".format(end_y))

                    # extract the face ROI and grab the ROI dimensions
                    face = img[start_y:end_y, start_x:end_x]

                    (fh, fw) = face.shape[:2]
                    # ensure the face width and height are sufficiently large
                    if fw < 20 or fh < 20:
                        pass
                    else:
                        faces_bb.append(box)

        if len(faces_bb) > 0:
            faces_bb = np.array(faces_bb)

        return faces_bb


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=50, thickness=(2, 2, 2)):
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = (
        size
        * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw))
        + tdy
    )

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = (
        size
        * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll))
        + tdy
    )

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), thickness[2])

    return img
