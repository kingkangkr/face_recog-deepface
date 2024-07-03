import cv2
import socket
import struct
import numpy as np
import face_recognition
import os
import time
import threading
from queue import Queue

def receive_frames(connection, frame_queue):
    data = b""
    payload_size = struct.calcsize(">L")
    while True:
        while len(data) < payload_size:
            data += connection.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += connection.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        img = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        frame_queue.put(img)

def process_frames(frame_queue, result_queue, encodeListKnown, classNames):
    frame_count = 0
    PROCESS_INTERVAL = 3  # 3프레임마다 처리
    last_face_locations = []
    last_face_names = []

    while True:
        img = frame_queue.get()
        frame_count += 1

        if frame_count % PROCESS_INTERVAL == 0:
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            last_face_locations = []
            last_face_names = []

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex] and faceDis[matchIndex] < 0.6:
                    name = classNames[matchIndex].upper()
                else:
                    name = 'UNKNOWN'

                # 스케일 조정된 좌표 저장
                last_face_locations.append((faceLoc[0]*4, faceLoc[1]*4, faceLoc[2]*4, faceLoc[3]*4))
                last_face_names.append(name)

        # 모든 프레임에 대해 마지막으로 인식된 얼굴 정보를 사용하여 표시
        for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        result_queue.put(img)



def display_frames(result_queue):
    while True:
        img = result_queue.get()
        cv2.imshow('Attendance System', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# 기존의 identifyEncodings 함수는 그대로 유지
def identifyEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('192.168.1.1', 8080))  # PC2의 실제 IP 주소로 변경
    server_socket.listen(1)
    connection, client_address = server_socket.accept()

    path = 'Attendance_data'
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = identifyEncodings(images)
    print('Encoding Complete')

    frame_queue = Queue(maxsize=10)
    result_queue = Queue(maxsize=10)

    receive_thread = threading.Thread(target=receive_frames, args=(connection, frame_queue))
    process_thread = threading.Thread(target=process_frames, args=(frame_queue, result_queue, encodeListKnown, classNames))
    display_thread = threading.Thread(target=display_frames, args=(result_queue,))

    receive_thread.start()
    process_thread.start()
    display_thread.start()

    receive_thread.join()
    process_thread.join()
    display_thread.join()

    connection.close()
    server_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


