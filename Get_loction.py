import cv2
import numpy as np
from number_recognition import CNN
import torch
from test_model import play_test
####################### 设置参数 #######################
widthImg = 640
heightImg = 480
kernal = np.ones((5, 5))
minArea = 400
maxArea = 99999999999
net = CNN(1, 10)
net.load_state_dict(torch.load('./models/model1.pth'))
cap = cv2.VideoCapture(0)
cap.set(3, widthImg)  # 设置参数，10为亮度
cap.set(4, heightImg)
cap.set(10, 150)
is_video = True


# 显示
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


# 预处理函数
def pre_process(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)  # 边沿检测
    imgDial = cv2.dilate(imgCanny, kernal, iterations=2)  # 膨胀
    imgThres = cv2.erode(imgDial, kernal, iterations=1)  # 腐蚀
    return imgThres, imgDial


# 预测
def predict(p_img):
    global net
    pic = cv2.resize(p_img, (28, 28), interpolation=cv2.INTER_CUBIC)
    im_data = np.array(pic)
    im_data = torch.from_numpy(im_data).float()
    im_data = im_data.view(1, 1, 28, 28)
    out = net(im_data)
    _, pred = torch.max(out, 1)
    return pred


# 画框
def draw_re(cnt):
    area = cv2.contourArea(cnt)
    # print(area)
    if maxArea > area > minArea:  # 面积大于minArea像素为封闭图形
        cv2.drawContours(imgCopy, cnt, -1, (255, 0, 0), 3)  # 不要在原图上面画，-1是所有的轮廓
        peri = cv2.arcLength(cnt, True)  # 计算周长
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 计算有多少个拐角
        x, y, w, h = cv2.boundingRect(approx)  # 得到外接矩形的大小
        a = (w + h) // 2
        dd = abs((w - h) // 2)  # 边框的差值
        # cv2.rectangle(imgContour,(x, y),(x+w,y+h),(255,255,255),2)    # 白色
        if w <= h:  # 得到一个正方形框，边界往外扩充10像素
            xx = x - dd - 10
            yy = y - 10
            ss = h + 20
            cv2.rectangle(imgCopy, (x - dd - 10, y - 10), (x + a + 10, y + h + 10), (0, 0, 255),
                          2)  # 看看框选的效果，在imgCopy中
            # print(a + dd, h)
        else:  # 边界往外扩充10像素值
            xx = x - 10
            yy = y - dd - 10
            ss = w + 20
            cv2.rectangle(imgCopy, (x - 10, y - dd - 10), (x + w + 10, y + a + 10), (0, 0, 255), 2)
            # print(a + dd, w)
        if x != 0 or y != 0:  # 图像不能为0
            return [xx, yy, ss]
        else:
            return [20, 20, 10]
    return []


# 找数字位置
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 检索外部轮廓
    x_list = np.array([])
    y_list = np.array([])
    s_list = np.array([])
    for cnt in contours:  # 每一个轮廓线
        loc = draw_re(cnt)
        cv2.drawContours(img, cnt, -1, (0, 0, 0), thickness=-1)
        if loc and loc[0] >= 10 and loc[1] >= 10 and loc[2] > 10:
            x_list = np.append(x_list, loc[0])
            y_list = np.append(y_list, loc[1])
            s_list = np.append(s_list, loc[2])
    return np.vstack((x_list, y_list, s_list)).T.astype(np.int16)


# 得到预测的数字，并显示
def detect_num(imgProcess, locations):
    global imgCopy
    print(locations)
    num_detected = np.array([])
    if locations.shape[0] >= 9:
        for loc in locations:
            imgRes = imgProcess[loc[1] - 5:loc[1] + loc[2] + 10, loc[0] - 5:loc[0] + loc[2] + 10]  # 得到数字区域图片,注意先是y，再是x
            pre = predict(imgRes)
            num_detected = np.append(num_detected, pre[0].item())
            cv2.putText(imgCopy, str(pre[0].item()), (loc[0], loc[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    return num_detected


# 得到八数码问题
def minor_problem(locations, num_detected):
    problem_list = np.zeros(9, dtype=np.int16)  # 八数码问题的np list
    problem_loc = np.array([])
    if num_detected.shape[0] == 9:
        # 第一行
        sort_index = np.argsort(locations[-3:], axis=0).T  # 转置一下
        problem_list[-9:-6] = num_detected[sort_index[0]+6]
        # 第二行
        sort_index = np.argsort(locations[-6:-3], axis=0).T  # 转置一下
        problem_list[-6:-3] = num_detected[sort_index[0]+3]
        # 第三行
        sort_index = np.argsort(locations[-9:-6], axis=0).T  # 转置一下
        problem_list[-3:] = num_detected[sort_index[0]]
        # 图片对应位置
        problem_loc = np.vstack([np.sort(locations[-3:], axis=0),
                                 np.sort(locations[-6:-3], axis=0),
                                 np.sort(locations[-9:-6], axis=0)])
        return problem_list, problem_loc
    return problem_list, problem_loc


if __name__ == "__main__":
    is_video = 0
    if is_video:
        while True:
            success, img = cap.read()
            imgCopy = img.copy()
            imgProcess, show2 = pre_process(img)  # 可以很好地把图像取出来
            locations = getContours(imgProcess)
            num_detected = detect_num(imgProcess, locations)
            problem_list, problem_loc = minor_problem(locations, num_detected)
            cv2.putText(imgCopy, str(problem_list), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            stackImg = stackImages(0.6, ([img, imgCopy], [show2, imgProcess]))
            cv2.imshow("Video", stackImg)
            k = cv2.waitKey(1)
            if k == 27:
                # 通过esc键退出摄像
                cv2.destroyAllWindows()
                break
            elif k == ord("s"):
                # 通过s键保存图片，并退出。
                play_test(problem_list, problem_loc, img)
                cv2.destroyAllWindows()
                break
    else:
        img = cv2.imread("./img/test1.png")
        imgCopy = img.copy()
        imgProcess, show2 = pre_process(img)  # 可以很好地把图像取出来
        locations = getContours(imgProcess)
        num_detected = detect_num(imgProcess, locations)
        problem_list, problem_loc = minor_problem(locations, num_detected)
        cv2.putText(imgCopy, str(problem_list), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        print(num_detected)
        stackImg = stackImages(0.5, ([img, imgCopy], [show2, imgProcess]))
        while True:
            cv2.imshow("Video", stackImg)
            k = cv2.waitKey(1)
            if k == 27:
                # 通过esc键退出摄像
                cv2.destroyAllWindows()
                break
            elif k == ord("s"):
                # 通过s键保存图片，并退出。
                play_test(problem_list, problem_loc, img)
                cv2.destroyAllWindows()
                break
