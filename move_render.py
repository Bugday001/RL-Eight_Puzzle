import cv2
import numpy as np
# from Get_loction import minor_problem

loc = np.array([[339, 341, 72],
                [138, 339, 71],
                [236, 337, 74],
                [336, 239, 89],
                [123, 240, 85],
                [231, 238, 88],
                [349, 142, 76],
                [234, 128, 96],
                [128, 124, 99]])

# 图片每个数字的位置从左到右
problem_loc = np.array([[128, 124, 76],
                        [234, 128, 96],
                        [349, 142, 99],
                        [123, 238, 85],
                        [231, 239, 88],
                        [336, 240, 89],
                        [138, 337, 71],
                        [236, 339, 72],
                        [339, 341, 74]])
num_list = np.array([3, 0, 8, 5, 7, 6, 2, 1, 4])  # num detect
# 上下左右四个移动方向
dRow = [-1, 1, 0, 0]
dCol = [0, 0, -1, 1]


class NumImg:
    # 原传入原始图片,原始问题,数字原始位置
    def __init__(self, origin_img, origin_list, origin_loc, row=3, col=3):
        self.s = np.max(origin_loc.T, 1)[2]  # 截取正方形边长
        self.origin_loc = origin_loc
        self.origin_img = origin_img.copy()
        self.img = origin_img.copy()
        self.origin_list = origin_list

    def img_move(self, state):
        for index, each in enumerate(state.flatten()):
            each_img_loc = self.origin_loc[np.where(self.origin_list == each)][0]
            self.img[self.origin_loc[index][1]:self.origin_loc[index][1] + self.s,self.origin_loc[index][0]:self.origin_loc[index][0] + self.s] =\
                self.origin_img[each_img_loc[1]:each_img_loc[1] + self.s, each_img_loc[0]:each_img_loc[0] + self.s]
        return self.img


def move(img, problem_list, direction=0):
    s = np.max(loc.T, 1)[2]
    zero_num_loc = np.where(problem_list == 0)  # 0位置
    switch_num_loc = (zero_num_loc[0] + dRow[direction], zero_num_loc[1] + dCol[direction])  # 与之交换的位置
    # 获取交换位置
    switch_num = problem_list[switch_num_loc]
    switch_img_loc = loc[np.where(num_list == switch_num)][0]
    # 获取0位置
    zero_img_loc = loc[np.where(num_list == 0)][0]
    # 交换
    # 交换图片
    temp_img = img[switch_img_loc[1]:switch_img_loc[1] + s, switch_img_loc[0]:switch_img_loc[0] + s].copy()
    img[switch_img_loc[1]:switch_img_loc[1] + s, switch_img_loc[0]:switch_img_loc[0] + s] = \
        img[zero_img_loc[1]:zero_img_loc[1] + s, zero_img_loc[0]:zero_img_loc[0] + s]
    img[zero_img_loc[1]:zero_img_loc[1] + s, zero_img_loc[0]:zero_img_loc[0] + s] = \
        temp_img
    # 交换位置数组
    temp_loc = zero_img_loc.copy()
    loc[np.where(num_list == 0)] = switch_img_loc
    loc[np.where(num_list == switch_num)] = temp_loc

    return img


if __name__ == "__main__":
    img = cv2.imread("./img/test1.png")
    # problem_list, problem_loc = minor_problem(loc, num_list)
    testimg = NumImg(img, problem_list, problem_loc)
    img = testimg.img_move([3, 0, 8, 5, 7, 6, 2, 1, 4])
    cv2.namedWindow('2', 0)
    cv2.resizeWindow('2', 640, 480)  # 自己设定窗口图片的大小
    cv2.imshow("2", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # #
    # s = np.max(loc.T, 1)[2]
    # # img = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # cv2.namedWindow('2', 0)
    # cv2.resizeWindow('2', 640, 480)  # 自己设定窗口图片的大小
    # cv2.imshow("2", img)
    # problem_list = minor_problem(loc, num_list).reshape(3, -1)
    # move(img, problem_list, 0)
    # # move(img, 1, problem_list)
    # cv2.namedWindow('findCorners', 0)
    # cv2.resizeWindow('findCorners', 640, 480)  # 自己设定窗口图片的大小
    # cv2.imshow("findCorners", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
