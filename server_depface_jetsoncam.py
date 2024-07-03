import cv2
import numpy as np
import face_recognition
import os
from deepface import DeepFace
import socket
import pickle
import struct

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
cap = cv2.VideoCapture(0)  # or the appropriate camera index

# Initialize variables
frame_count = 0
PROCESS_INTERVAL = 2  # Process every 3 frames
last_face_info = []

# Socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.1.1', 8080))  # Listen on all interfaces, port 8000
server_socket.listen(1)
print("Waiting for connection...")
client_socket, addr = server_socket.accept()
print(f"Connected to {addr}")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        break

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
            try:
                emotion_results = DeepFace.analyze(img_path=face_img, actions=['emotion'], enforce_detection=False)
                emotion_text = emotion_results[0]["dominant_emotion"] if emotion_results else ''
            except Exception as e:
                print(f"Error in emotion analysis: {e}")
                emotion_text = ''

            last_face_info.append((name, (x1, y1, x2, y2), emotion_text))

    # Draw rectangles and text for all detected faces
    for name, (x1, y1, x2, y2), emotion_text in last_face_info:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        if emotion_text:
            cv2.putText(img, emotion_text, (x1 + 6, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Send the processed image to the client
    _, buffer = cv2.imencode('.jpg', img)
    data = pickle.dumps(buffer)
    message_size = struct.pack("L", len(data))
    try:
        client_socket.sendall(message_size + data)
    except Exception as e:
        print(f"Error sending data: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client_socket.close()
server_socket.close()
