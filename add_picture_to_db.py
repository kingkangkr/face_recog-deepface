import cv2
import time
from PIL import Image
import face_recognition
import os

def capture_image(camera_id=None):
    """
    카메라를 사용하여 이미지를 캡처합니다.

    args:
    camera_id : int
    """
    path = "Attendance_data/"
    if not os.path.exists(path):
        os.makedirs(path)

    if camera_id is None:
        camera_id = 0  # 기본 카메라 ID

    Name = input("Please Enter your name: ")
    file_path = f'{path}{Name}.png'

    camera = cv2.VideoCapture(camera_id)

    # 5초 동안 카운트다운 및 카메라 화면 표시
    for i in range(5, 0, -1):
        return_value, frame = camera.read()
        if return_value:
            cv2.imshow('Camera', frame)
            print(i)
            cv2.waitKey(1000)  # 1초 대기
        else:
            print("Failed to capture image during countdown")
            camera.release()
            cv2.destroyAllWindows()
            return None

    # 최종 이미지 캡처
    return_value, image = camera.read()
    if return_value:
        cv2.imwrite(file_path, image)
        print(f"Image saved at {file_path}")
    else:
        print("Failed to capture image")
        camera.release()
        cv2.destroyAllWindows()
        return None

    camera.release()
    cv2.destroyAllWindows()
    return file_path

def find_faces_in_image(image_path):
    """
    이미지에서 얼굴을 찾아서 표시합니다.

    args:
    image_path : str
    """
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    print(f"I found {len(face_locations)} face(s) in this photograph.")

    for i, face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        face_file_path = f"{os.path.splitext(image_path)[0]}_face_{i+1}.png"
        pil_image.save(image_path)
        print(f"Face image saved at {face_file_path}")
        pil_image.show()

def main():
    image_path = capture_image()
    if image_path:
        find_faces_in_image(image_path)

if __name__ == "__main__":
    main()
