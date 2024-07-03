import cv2
import numpy as np
import face_recognition
import os
from deepface import DeepFace

def identifyEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Preprocessing the data 
path = 'Attendance_data'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Encoding of input image data
encodeListKnown = identifyEncodings(images)
print('Encoding Complete')

# Camera capture 
cap = cv2.VideoCapture('/dev/video0')

# Initialize variables
frame_count = 0
PROCESS_INTERVAL = 2  # 3프레임마다 처리
last_face_info = []

while True:
    success, img = cap.read()
    frame_count += 1

    if frame_count % PROCESS_INTERVAL == 0:
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        last_face_info = []

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            threshold = 0.6

            if matches[matchIndex] and faceDis[matchIndex] < threshold:
                name = classNames[matchIndex].upper()
            else:
                name = 'UNKNOWN'

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            face_img = img[y1:y2, x1:x2]
            emotion_results = DeepFace.analyze(img_path=face_img, actions=['emotion'], enforce_detection=False)

            emotion_text = ''
            if len(emotion_results) > 0:
                emotion = emotion_results[0]["dominant_emotion"]
                emotion_text = str(emotion)

            last_face_info.append((name, (x1, y1, x2, y2), emotion_text))

    for name, (x1, y1, x2, y2), emotion_text in last_face_info:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        if emotion_text:
            cv2.putText(img, emotion_text, (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
