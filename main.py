import face_recognition
from datetime import datetime
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
from ourfirebase import send_to_database
import pyrebase
import pandas as pd
from firebase import firebase


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

firebaseConfig = {
  "apiKey": "AIzaSyAOH4rcM94-kdi88o08nLCVu8QL6eGdo8o",
  "authDomain": "driverfinalv2.firebaseapp.com",
  "databaseURL": "https://driverfinalv2-default-rtdb.firebaseio.com",
  "projectId": "driverfinalv2",
  "storageBucket": "driverfinalv2.appspot.com",
  "messagingSenderId": "635919295800",
  "appId": "1:635919295800:web:0ef07dd405fd71ce5f8dcd"
}

my_firebase = pyrebase.initialize_app(firebaseConfig)
storage = my_firebase.storage()
user_url = "https://driverfinalv2-default-rtdb.firebaseio.com/"
from firebase import firebase
firebase = firebase.FirebaseApplication(user_url)

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 5
YAWN_THRESH = 22
# alarm_status = False
# alarm_status2 = False
# saying = False
COUNTER = 0


# def alarm(msg):
#     global alarm_status
#     global alarm_status2
#     global saying

    # while alarm_status:
    #     print('call')
    #     s = 'espeak "'+msg+'"'
    #     print('S: ',s)
    #     os.system("echo 'hello world'")
        #os.system(s)

#     if alarm_status2:
#         print('call')
#         saying = True
#         s = 'espeak "' + msg + '"'
#         os.system("echo 'hello world'")
# #        os.system(s)
#         saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


# Face Detection & Recognition
path = 'ImagesAttendance'
images = []
driversnames = []     #names

myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    driversnames.append(os.path.splitext(cl)[0].split("_")[0])
print(driversnames)

def screenshot():
    global cap
    # cv2.imshow("screenshot", cam.read()[1]) # shows the screenshot directly
    cv2.imwrite('C:/Users/Mostafa/PycharmProjects/Face_Rek/CarLogs/Unauthorized User/unauthorized_user.png',cap.read()[1]) # or saves it to disk
    storage.child("CarLogs/Unauthorized User/unauthorized_user.png").put("CarLogs/Unauthorized User/unauthorized_user.png")
    url = storage.child("CarLogs/Unauthorized User/unauthorized_user.png").get_url("itwVefusLWh0R6ifPC3AANyOtQg1")
    firebase.put("/Employees/itwVefusLWh0R6ifPC3AANyOtQg1", "unauthorized user", url)


#encoding process
def findEncodings(images):
    encodeList = []
    for img in images:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        #print('encode', encode)
        encodeList.append(encode)
    return encodeList
encodelistknown = findEncodings(images)
print('encoding complete')

def driversheet(user):
    df = pd.read_csv('CarLogs/Drivers Sheet.csv')
    users_list = ['MOSTAFA']
    if (user in users_list):
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        df.Time[users_list.index(user)] = dtString
    else:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        df.Time[df.index[-1]] = dtString
    df.to_csv("CarLogs/Drivers Sheet.csv", index=False)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
Flag=True
counter=0
counter_2=0
auth_flag = 0
while Flag:

    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name=driversnames[matchIndex].upper()
            driversheet(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            # send(name)
            if (name == 'MOSTAFA'):
              counter=counter+1
              if(counter>10):
                  Flag=False
                  print('Welcome:',name)
                  print('Access Granted')
                  cv2.destroyAllWindows()
        else:
            name = 'Not Registered'
            driversheet(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            counter_2 = counter_2 + 1
            if (counter_2 > 10):
                Flag = False
                auth_flag = 1
                cv2.destroyAllWindows()
            # screenshot()
            # break

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# markAttendance(name)
storage.child("CarLogs/Drivers Sheet.csv").put("CarLogs/Drivers Sheet.csv")
url = storage.child("CarLogs/Drivers Sheet.csv").get_url("itwVefusLWh0R6ifPC3AANyOtQg1")
firebase.put("/Employees/itwVefusLWh0R6ifPC3AANyOtQg1", "drivers_sheet", url)
if(auth_flag == 1):
    screenshot()
cap.release()
cv2.destroyAllWindows()
if(auth_flag == 0):
    print('Drowsiness Part: ')

    print("-> Loading the predictor and detector...")
    # detector = dlib.get_frontal_face_detector()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # Faster but less accurate
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    print("-> Starting Video Stream")
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # rects = detector(gray, 0)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)


        # for rect in rects:
        for (x, y, w, h) in rects:
            # from firebase import firebase
            # url="https://driverfinalv2-default-rtdb.firebaseio.com/"
            # firebase = firebase.FirebaseApplication(url)
            # firebase.put("/Employees/5zZOEcaTbWc5qQKss9MY3UIh2y43", "drowsy",0)
            myflag = False

            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)
            # print('1')
            # print('Eye:',EYE_AR_THRESH)
            # print('Ear:',ear)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # print('2')
                # print('COUNTER:',COUNTER)
                # print('Eye 2:', EYE_AR_CONSEC_FRAMES)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # print('3')
                    # if alarm_status == False:
                    #     alarm_status = True
                    #     t = Thread(target=alarm, args=('wake up sir',))
                    #     t.deamon = True
                    #     t.start()
                    # print('4')
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # print('5')
                    myflag=True

            else:
                # print('6')
                COUNTER = 0
                # alarm_status = False

            if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #     if alarm_status2 == False and saying == False:
            #         alarm_status2 = True
            #         t = Thread(target=alarm, args=('take some fresh air sir',))
            #         t.deamon = True
            #         t.start()
            # else:
            #     alarm_status2 = False

            cv2.putText(frame, "Eye: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # print('7')
            send_to_database(myflag)
    #         if(myflag == True):
    #             from firebase import firebase
    #             url="https://driverfinalv2-default-rtdb.firebaseio.com/"
    #             firebase = firebase.FirebaseApplication(url)
    #             firebase.put("/Employees/5zZOEcaTbWc5qQKss9MY3UIh2y43", "drowsy",1)
    #         else:
    #             from firebase import firebase
    #             url="https://driverfinalv2-default-rtdb.firebaseio.com/"
    #             firebase = firebase.FirebaseApplication(url)
    #             firebase.put("/Employees/5zZOEcaTbWc5qQKss9MY3UIh2y43", "drowsy",0)


    #         firebase.put("/Employees/5zZOEcaTbWc5qQKss9MY3UIh2y43", "drowsy",0)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

# def alarm(msg):
#     global alarm_status
#     global alarm_status2
#     global saying

#     while alarm_status:
#         print('call')
#         s = 'espeak "'+msg+'"'
#         print('S: ',s)
#         os.system("echo 'hello world'")
#         #os.system(s)

#     if alarm_status2:
#         print('call')
#         saying = True
#         s = 'espeak "' + msg + '"'
#         os.system("echo 'hello world'")
# #        os.system(s)
#         saying = False

# def eye_aspect_ratio(eye):
#     A = dist.euclidean(eye[1], eye[5])
#     B = dist.euclidean(eye[2], eye[4])

#     C = dist.euclidean(eye[0], eye[3])

#     ear = (A + B) / (2.0 * C)

#     return ear

# def final_ear(shape):
#     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#     leftEye = shape[lStart:lEnd]
#     rightEye = shape[rStart:rEnd]

#     leftEAR = eye_aspect_ratio(leftEye)
#     rightEAR = eye_aspect_ratio(rightEye)

#     ear = (leftEAR + rightEAR) / 2.0
#     return (ear, leftEye, rightEye)

# def lip_distance(shape):
#     top_lip = shape[50:53]
#     top_lip = np.concatenate((top_lip, shape[61:64]))

#     low_lip = shape[56:59]
#     low_lip = np.concatenate((low_lip, shape[65:68]))

#     top_mean = np.mean(top_lip, axis=0)
#     low_mean = np.mean(low_lip, axis=0)

#     distance = abs(top_mean[1] - low_mean[1])
#     return distance


# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--webcam", type=int, default=0,
#                 help="index of webcam on system")
# args = vars(ap.parse_args())

# EYE_AR_THRESH = 0.3
# EYE_AR_CONSEC_FRAMES = 30
# YAWN_THRESH = 20
# alarm_status = False
# alarm_status2 = False
# saying = False
# COUNTER = 0

# print("-> Loading the predictor and detector...")
# #detector = dlib.get_frontal_face_detector()
# detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# print("-> Starting Video Stream")
# vs = VideoStream(src=args["webcam"]).start()
# #vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
# time.sleep(1.0)

# while True:

#     frame = vs.read()
#     frame = imutils.resize(frame, width=450)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     #rects = detector(gray, 0)
#     rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
# 		minNeighbors=5, minSize=(30, 30),
# 		flags=cv2.CASCADE_SCALE_IMAGE)

#     #for rect in rects:
#     for (x, y, w, h) in rects:
#         rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)

#         eye = final_ear(shape)
#         ear = eye[0]
#         leftEye = eye [1]
#         rightEye = eye[2]

#         distance = lip_distance(shape)

#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         lip = shape[48:60]
#         cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

#         if ear < EYE_AR_THRESH:
#             COUNTER += 1

#             if COUNTER >= EYE_AR_CONSEC_FRAMES:
#                 if alarm_status == False:
#                     alarm_status = True
#                     t = Thread(target=alarm, args=('wake up sir',))
#                     t.deamon = True
#                     t.start()

#                 cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         else:
#             COUNTER = 0
#             alarm_status = False

#         if (distance > YAWN_THRESH):
#                 cv2.putText(frame, "Yawn Alert", (150, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 if alarm_status2 == False and saying == False:
#                     alarm_status2 = True
#                     t = Thread(target=alarm, args=('take some fresh air sir',))
#                     t.deamon = True
#                     t.start()
#         else:
#             alarm_status2 = False

#         cv2.putText(frame, "Eye: {:.2f}".format(ear), (300, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break