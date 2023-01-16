import cv2
import numpy as np
from pathlib import Path

WIN_NAME_FULL = "full"
WIN_NAME_ZOOM = "zoom"

def main():
    video_path = Path("/home/oba/data/printer_film_short.mp4")
    if not video_path.exists():
        raise ValueError("The video file does not exist: " + str(video_path))

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError("Could not open video :(")

    cv2.namedWindow(WIN_NAME_FULL)

    ret, frame = video.read()
    cv2.imshow(WIN_NAME_FULL, frame)

    while cv2.waitKey(0):
        ret, frame = video.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, 1000, 0.1, 10)
        for corner in corners.squeeze().astype(int):
            cv2.circle(frame, corner, 3, (255, 0, 255))
        cv2.imshow(WIN_NAME_FULL, frame)


if __name__ == '__main__':
    main()
