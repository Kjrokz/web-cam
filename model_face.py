from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mtcnn = MTCNN(
    image_size=240, margin=0, keep_all=True, min_face_size=40
)  # keep_all=True
resnet = InceptionResnetV1(pretrained="vggface2").eval()


load_data = torch.load("models/caras.pt")
embedding_list = load_data[0]
name_list = load_data[1]


def model_face(frame):
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                dist_list = (
                    []
                )  # list of matched distances, minimum distance is used to identify the person

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                    min_dist = min(dist_list)  # get minumum dist value
                    min_dist_idx = dist_list.index(min_dist)  # get minumum dist index
                    name = name_list[
                        min_dist_idx
                    ]  # get name corrosponding to minimum dist

                    box = boxes[i]
                    ##original_frame = frame.copy() # storing copy of frame before drawing on it
                    ##print(min_dist)
                    if min_dist < 0.90:
                        frame = cv2.putText(
                            frame,
                            name,
                            (int(box[0]), int(box[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA,
                        )

                        frame = cv2.rectangle(
                            frame,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (255, 0, 0),
                            2,
                        )
                        ##model_eye(frame)
                        # model_eye_blinks(frame,total,counter)
