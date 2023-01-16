import cv2
import numpy as np
from pathlib import Path


def main():
    video_path = Path("/home/oba/data/printer_film.mp4")
    if not video_path.exists():
        raise ValueError("The video file does not exist: " + str(video_path))

    video = cv2.VideoCapture(str(video_path))

    if not video.isOpened():
        raise ValueError("Could not open video :(")

    cv2.namedWindow("hallo")

    while cv2.waitKey(0):
        ret, frame = video.read()
        cv2.imshow("hallo", frame)
        # corners = cv2.goodFeaturesToTrack(image, 10)


if __name__ == '__main__':
    main()
