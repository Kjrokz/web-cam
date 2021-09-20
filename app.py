from model_direction import direction
from model_eye import model_eye
import torch
from model_eye_blinks import model_eye_blinks
from flask import *
import cv2
from model_face import model_face
from datetime import timedelta
import os
import base64
import numpy as np
from flask_cors import CORS
import constants


##from waitress import serve


app = Flask(__name__)
CORS(app)

#app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=1)
ALLOWED_EXTENSIONS = set(["png", "jpg"])
UPLOAD_FOLDER = r"./uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

#@app.after_request
#def after_request(response):
    #response.headers["Access-Control-Allow-Origin"] = "*"
    #response.headers["Access-Control-Allow-Credentials"] = "true"
    #response.headers["Access-Control-Allow-Methods"] = "POST"
    ##response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    #return response

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS

@app.route("/upload",methods=["GET","POST"])
def gen_frames():  # generate frame by frame from camera

    #print(request.get_json())
    image = request.get_json()
    #print(image["image"])
    uri = image["image"]

    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    #frame = cv2.imread(img)
        #print(frame)
    model_face(frame)
    model_eye_blinks(frame)
    direction(frame)
    ret, buffer = cv2.imencode(".jpg", frame)
    frame = base64.b64encode(buffer)
        #frame = buffer.tobytes()
        #print(ret)
        #print(buffer)
        #print(frame)

        #return(
        #b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        #)  # concat frame one by one and show result
    return frame
    #return img


    ##return jsonify({"1":1})

@app.route("/facial",methods=["GET","POST"])
def gen_facial():  # generate frame by frame from camera

    #print(request.get_json())
    image = request.get_json()
    #print(image["image"])
    uri = image["image"]

    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    model_face(frame)
   
    ret, buffer = cv2.imencode(".jpg", frame)
    frame = base64.b64encode(buffer)

    return frame

@app.route("/eyes",methods=["GET","POST"])
def gen_eyes():  # generate frame by frame from camera

    #print(request.get_json())
    image = request.get_json()
    #print(image["image"])
    uri = image["image"]

    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    model_eye_blinks(frame)
   
    ret, buffer = cv2.imencode(".jpg", frame)
    frame = base64.b64encode(buffer)

    return frame

@app.route("/direction",methods=["GET","POST"])
def gen_direction():  # generate frame by frame from camera

    #print(request.get_json())
    image = request.get_json()
    #print(image["image"])
    uri = image["image"]

    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    direction(frame)
   
    ret, buffer = cv2.imencode(".jpg", frame)
    frame = base64.b64encode(buffer)

    return frame
    
    #frame = cv2.flip(frame, 1)
    #model_face(frame)
    #model_eye_blinks(frame)
    #direction(frame)
        ##model_eye(frame)
    #ret, buffer = cv2.imencode(".jpg", frame)
    #frame = buffer.tobytes()
    #yield (
        #b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    #)  # concat frame one by one and show result
    
@app.route("/reset")
def reset():
    constants.HEAD = 0
    constants.MAS = 0
    constants.COUTER = 0
    constants.TOTAL = 0
    return jsonify({"status":1})


@app.route("/video_feed")
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
    ##serve(app, host="0.0.0.0", port=8000)
