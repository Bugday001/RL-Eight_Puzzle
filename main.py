import numpy as np
import cv2
from number_recognition import CNN
from Get_loction import detect_num, stackImages


if __name__ == "__name__":
    print("测试用！")
    print("执行Get_loction.py开始！")
    print("出现检测界面，在检测界面按`s`即可开始求解八数码。")
    # img = cv2.imread("./img/test1.jpg")
    # imgProcess, imgRes = detect_num(img)
    # cv2.imshow("Video", imgRes)
    # cv2.waitKey(0)