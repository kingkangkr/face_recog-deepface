import cv2
import numpy as np
import face_recognition
import os
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
cap = cv2.VideoCapture('/dev/video0')

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

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.6:
            name = classNames[matchIndex].upper()
        else:
            name = 'UNKNOWN'

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

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
