import cv2
import matplotlib.pyplot as plt

faceModel='opencv_face_detector_uint8.pb'
faceconfig='opencv_face_detector.pbtxt'

ageModel='age_net.caffemodel'
ageConfig='age_deploy.prototxt'

genderModel='gender_net.caffemodel'
genderConfig='gender_deploy.prototxt'

faceNet=cv2.dnn.readNet(faceModel, faceconfig)
ageNet=cv2.dnn.readNet(ageModel, ageConfig)
genderNet=cv2.dnn.readNet(genderModel, genderConfig)
