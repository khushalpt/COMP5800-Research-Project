import os
import time
import glob
import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import TruncatedSVD
from flask import Flask, render_template, Response, jsonify, request
import face_recognition
import imutils

DATASET = 'data/webapp'
width = height = 64

def get_model():
    n_imgs = 0
    for root, dirs,files in os.walk(DATASET):
        for file in files:
            if (file.endswith(".jpg") or file.endswith(".png")) and len(os.listdir(root))>0:
                n_imgs += 1
    X_train = np.zeros((n_imgs, width, height, ), dtype=np.float32)
    X_train_hog = np.zeros((n_imgs, 128 ), dtype=np.float32)
    y_train = np.zeros((n_imgs, ))
    i = 1
    unique_labels = {}
    id_to_name = {}
    c = 0
    for root, dirs,files in os.walk(DATASET):
        for file in files:
            if (file.endswith(".jpg") or file.endswith(".png")) and len(os.listdir(root))>0:
                img_path = os.path.join(root, file)
                img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_rgb = imutils.resize(img_rgb, width=128)
                img = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY), (width, height))
                label = root.split('/')[-1]
                if label not in unique_labels:
                    unique_labels[label] = c
                    id_to_name[c] = label
                    c+=1
                X_train[i-1] = np.array(img)
                boxes = face_recognition.face_locations(img_rgb, model="hog")
                encodings = face_recognition.face_encodings(img_rgb, boxes)
                if len(encodings)>0:
                    X_train_hog[i-1]=encodings[0]
                y_train[i-1] = unique_labels[label]
                i+=1

    X_train = X_train.reshape(-1,width*height)

    svdmodel = TruncatedSVD(n_components=128, random_state=0)
    X_train_svd = svdmodel.fit_transform(X_train)

    ldamodel = LinearDiscriminantAnalysis()
    ldamodel.fit(X_train_hog, y_train)
    return id_to_name, svdmodel, ldamodel

app = Flask(__name__)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.id_to_name, self.svdmodel, self.ldamodel = get_model()
        self.last_pred = "initializing..."

    def reinit(self):
        self.id_to_name, self.svdmodel, self.ldamodel = get_model()

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        img = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           
        img_rgb = imutils.resize(img_rgb, width=128)
        img = np.array(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (64,64)))
        img = img.reshape(-1, 64*64)
        img_svd = self.svdmodel.transform(img)

        boxes = face_recognition.face_locations(img_rgb, model="hog")
        encodings = face_recognition.face_encodings(img_rgb, boxes)
        encoding = np.zeros((1,128))
        if len(encodings)>0:
            encoding[0] = encodings[0]
#        y_pred = self.ldamodel.predict(np.concatenate((img, encoding, img_svd), axis=1))
        y_pred = self.ldamodel.predict(encoding)
        self.last_pred = self.id_to_name[int(y_pred)] if len(boxes)>0 else "No face detected"
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


video_stream = VideoCamera()

@app.route('/')
def index():
    return render_template('./index.html')

def vid_gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# The live show of webcam feed
@app.route('/video_feed')
def video_feed():
    return Response(vid_gen(video_stream),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# The live show of the last predicted person
@app.route('/pred_feed')
def pred_feed():
    def generate(camera):
        yield camera.last_pred
    return Response(generate(video_stream), mimetype='text') 

# Upload a new face picture
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      name = request.form['name']
      if not os.path.exists(os.path.join(DATASET, name)):
          os.makedirs(os.path.join(DATASET, name))
      f.save(os.path.join(DATASET, name, f.filename))
      video_stream.reinit()
      return render_template('./index.html')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True,port="5000")
