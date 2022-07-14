import numpy as np
import cv2
from number_recognition import CNN
from Get_loction import detect_num, stackImages


if __name__ == "__name__":
    img = cv2.imread("./img/test1.jpg")
    imgProcess, imgRes = detect_num(img)
    cv2.imshow("Video", imgRes)
    cv2.waitKey(0)