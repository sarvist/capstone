import cv2 as cv

webcam_id = 0
found_cam = False
while webcam_id < 1000:
    cap = cv.VideoCapture(webcam_id)
    if cap.isOpened():
        break
    webcam_id += 1
print(webcam_id)
