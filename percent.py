import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pytz
import time

def identifyEncodings(images):
    '''Encoding is Recognition and comparing particular face in database or stored folder
    args:
    images:str
    '''
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
# split the data vk.png to vk
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

# Initialize frame timing variables
fps_start_time = 0
fps = 0

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Face recognition using dlib
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # 거리 값을 신뢰도로 변환
        threshold = 0.6
        confidence = (1 - faceDis[matchIndex]) * 100

        if matches[matchIndex] and faceDis[matchIndex] < threshold:
            name = classNames[matchIndex].upper()
            confidence_text = f'{confidence:.2f}%'
        else:
            name = 'UNKNOWN'
            confidence_text = '0.00%'

        print(name, confidence_text)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 45), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x1 + 6, y2 - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Calculate FPS
    fps_end_time = time.time()
    time_diff = fps_end_time - fps_start_time
    fps = 1 / time_diff
    fps_start_time = fps_end_time

    # Display FPS on the frame
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
