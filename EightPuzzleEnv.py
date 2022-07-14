# coding:UTF-8
# 八数码问题的环境
# import gym
import numpy as np
# from sklearn.model_selection import train_test_split
from ReversePairs import reversePairs_mergeSort
# from TestCase import testCase
import cv2
import time

# import seaborn as sns

# sns.set_style('whitegrid')
np.set_printoptions(suppress=True, linewidth=500, edgeitems=8, precision=4)

# 上下左右四个移动方向
dRow = [-1, 1, 0, 0]
dCol = [0, 0, -1, 1]


def swap(matrix, row1, col1, row2, col2):
    '''
       交换矩阵的两个元素
    '''
    t = matrix[row1, col1]
    matrix[row1, col1] = matrix[row2, col2]
    matrix[row2, col2] = t


def checkBounds(i, j, m, n):
    '''
        检测下标是否越界
    '''
    if i >= 0 and i < m and j >= 0 and j < n:
        return True
    else:
        return False


def arrayToMatrix(arr, m, n):
    '''
    数组转矩阵
    '''
    return np.resize(arr, [m, n])


def matirxToArray(matrix):
    '''
    矩阵转数组
    '''
    return matrix.ravel()


def checkReversePair(lst):
    '''
    检测逆序对数是否是偶数，注意去除0
    '''
    array = []
    for key in lst:
        if key != 0:
            array.append(key)
    rCnt = reversePairs_mergeSort(array)
    return rCnt & 1 == 0


def initBoard(m, n, env):
    '''
    初始化一个随机棋盘
    '''
    MAXITER = 65535
    for _ in range(MAXITER):
        perm = np.arange(m * n)
        p = np.random.random()
        # 随机选择sklearn或者numpy生成一个排列
        # if p <= 0.5:
        #     perm, _ = train_test_split(perm, test_size=0)
        # else:
        np.random.shuffle(perm)
        # 检测逆序对的奇偶性
        flag = checkReversePair(perm.copy())
        if flag and not np.array_equal(perm, env.target):
            return arrayToMatrix(perm, m, n)
        else:
            continue


def findZeroPos(board, m, n):
    '''
    找到0所在位置
    '''
    startRow = -1
    startCol = -1
    flag = True
    for i in range(0, m):
        if not flag:
            break
        for j in range(0, n):
            if board[i, j] == 0:
                startRow = i
                startCol = j
                flag = False
    return np.array([startRow, startCol])


def A_star_dist(target, now):
    '''
    A*估价,比较当前棋盘与最终棋盘，计算有多少棋子未摆放正确
    '''
    length = len(now)
    cnt = 0
    for i in range(0, length):
        if target[i] != now[i]:
            cnt += 1
    return cnt


class EightPuzzleEnv:  # gym.Env
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }
    # 动作空间
    ActionDim = 4
    # 默认行数
    DefaultRow = 2
    # 默认列数
    DefaultCol = 3
    # 成功回报
    SuccessReward = 10.0

    def __init__(self, m=DefaultRow, n=DefaultCol):
        self.viewer = None
        # 一轮episode最多执行多少次step
        if m == 2:
            self._max_episode_steps = 100  # 2048
        else:
            self._max_episode_steps = 30
        target = np.zeros([m, n], dtype=np.int16)
        for i in range(0, m):
            for j in range(0, n):
                target[i, j] = i * n + j + 1
        self.target = target.ravel()
        target[m - 1][n - 1] = 0
        self.m = m
        self.n = n
        self.last_step = -1

    def reset(self):
        '''
          重置为随机棋盘
        '''
        self.state = initBoard(self.m, self.n, self)
        self.pos = findZeroPos(self.state, self.m, self.n)
        return self.state

    def reset2(self, board):
        """
                重置为指定的棋盘，用于测试阶段
        """
        self.state = board.copy()
        self.pos = findZeroPos(self.state, self.m, self.n)
        return self.state

    def reset3(self, steps=4):
        # target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        self.state = self.target.copy().reshape(self.m, self.n)
        self.pos = findZeroPos(self.state, self.m, self.n)
        a = 0
        last_a = -1
        for setp in range(steps):
            nowRow = self.pos[0]
            nowCol = self.pos[1]
            a = np.random.randint(4)
            nextRow = nowRow + dRow[a]
            nextCol = nowCol + dCol[a]
            while (not checkBounds(nextRow, nextCol, self.m, self.n)) or\
                    (a == 0 and last_a == 1) or (a == 1 and last_a == 0) or\
                    (a == 2 and last_a == 3) or (a == 3 and last_a == 2):
                a = np.random.randint(4)
                nextRow = nowRow + dRow[a]
                nextCol = nowCol + dCol[a]
            last_a = a
            nextState = self.state.copy()
            # 移动方格
            swap(nextState, nowRow, nowCol, nextRow, nextCol)
            self.pos = np.array([nextRow, nextCol])
            self.state = nextState
        return self.state

    def reset_img(self, board, loc):
        """
        针对图片render的reset,传入棋盘和图片位置
        """
        self.state = board.copy()
        self.img_loc = loc.copy()
        self.pos = findZeroPos(self.state, self.m, self.n)
        return self.state

    def step(self, a):
        nowRow = self.pos[0]
        nowCol = self.pos[1]
        nextRow = nowRow + dRow[a]
        nextCol = nowCol + dCol[a]
        nextState = self.state.copy()
        # 检查越界
        if not checkBounds(nextRow, nextCol, self.m, self.n):
            return self.state, -2.0, False, {'info': -1, 'MSG': 'OutOfBounds!'}
        # 移动方格
        swap(nextState, nowRow, nowCol, nextRow, nextCol)
        self.pos = np.array([nextRow, nextCol])
        # 获得奖励
        re = self.reward(self.state, nextState, a)
        self.last_step = a
        self.state = nextState
        if re == EightPuzzleEnv.SuccessReward:
            return self.state, re, True, {'info': 2, 'MSG': 'Finish!'}
        return self.state, re, False, {'info': 1, 'MSG': 'NormalMove!'}

    def isFinish(self, s):
        '''
        检查是否到达终点
        '''
        if np.array_equal(s.ravel(), self.target):
            return True
        else:
            return False

    def reward(self, nowState, nextState, a):
        re = 0
        '''
        奖励函数
        '''
        if self.isFinish(nextState):
            # 到达终点，给予最大奖励
            return EightPuzzleEnv.SuccessReward
        else:
            # 对移动前的棋盘、移动后的棋盘分别进行估价A_star_dist
            # lastDist = Manhattan(self.target.reshape(self.m, self.n), nowState)
            # nowDist = Manhattan(self.target.reshape(self.m, self.n), nextState)
            lastDist = A_star_dist(self.target, nowState.ravel())
            nowDist = A_star_dist(self.target, nextState.ravel())
            if (a == 0 and self.last_step == 1) or (a == 1 and self.last_step == 0) or\
                    (a == 2 and self.last_step == 3) or (a == 3 and self.last_step == 2):
                re -= 0.5
            # 距离减小，给予较小惩罚
            if nowDist < lastDist:
                re -= 0.1
            # 距离不变，给予中等惩罚
            elif nowDist == lastDist:
                re -= 0.2
            # 距离增大，给予较大惩罚
            else:
                re -= 0.5
            return re

    def render(self, mode='human', close=False):
        '''
                渲染
        '''
        time.sleep(1)
        print('----------')
        for i in range(0, self.m):
            print('|', end='')
            for j in range(0, self.n):
                print(self.state[i][j], end='')
                print('|', end='')
            print()
        print('----------')

    def img_render(self, num_img):
        """
                图片渲染
        """
        time.sleep(0.1)
        img = num_img.img_move(self.state)
        cv2.imshow("2", img)
        cv2.waitKey(1000)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps


def Manhattan(target, state):
    """
    曼哈顿距离
    :return:
    """
    dist = 0
    for num in range(9):
        t = np.array(np.where(target == num))
        st = np.array(np.where(state == num))
        dist += np.sum(abs(t - st))
    return dist


def expDist(x, mu):
    '''
    指数分布
    均值和标准差均是:1/λ
    '''
    landa = 1 / mu
    return landa * np.exp(-landa * x)


def NDist(x, mu, std):
    '''
    正态分布
    '''
    par = 1 / (np.sqrt(2 * np.pi) * std)
    return par * np.exp(-(x - mu) ** 2 / 2 / std ** 2)


def buddaAgent(env):
    """
         佛系agent
    """
    return np.random.randint(0, env.ActionDim)


if __name__ == '__main__':
    env = EightPuzzleEnv(3, 3)
    env.reset3(8)
    env.reset3()
    env.reset3()
    # env = EightPuzzleEnv(2, 3)
    # # 统计到达终点的步数，猜测数据分布
    # lst = []
    # MAX_ITER = 2048
    # for i in range(MAX_ITER):
    #     env.reset()
    #     cnt = 0
    #     while True:
    #         a = buddaAgent(env)
    #         nextS, r, terminal, info = env.step(a)
    #         cnt += 1
    #         if terminal:
    #             break
    #     if i % 50 == 0:
    #         print(i, 'stepCnt:', cnt)
    #     lst.append(cnt)
    # print()
    # print('min:', np.min(lst))
    # print('mean:', np.mean(lst))
    # print('median:', np.median(lst))
    # print('max:', np.max(lst))
    # print('std:', np.std(lst))
    # print('mean divide std:', np.mean(lst) / np.std(lst))
    #
    # print('猜测数据服从指数分布')
    # max_value = int(np.max(np.max(lst)))
    # XTick = np.linspace(0, 2e4, 21)
    # plt.hist(lst, bins=64, alpha=0.5, color='red', edgecolor='red', density=True, range=(0, max_value))
    # x = np.linspace(0, max_value, max_value)
    # y = expDist(x, np.mean(lst))
    # plt.xticks(XTick)
    # plt.plot(x, y, color='blue', lw=2)
    #
    # # 对佛系agent进行测试
    # x, label = testCase()
    # okCnt = 0
    # for i in range(len(x)):
    #     board = x[i]
    #     correctCnt = label[i]
    #     env.reset()
    #     cnt = 0
    #     while True:
    #         a = buddaAgent(env)
    #         nextS, r, terminal, info = env.step(a)
    #         cnt += 1
    #         if terminal:
    #             break
    #     if cnt == correctCnt:
    #         okCnt += 1
    # print()
    # print('佛系智能体的准确率:{:.1f}%'.format(okCnt / len(x) * 100))
    # plt.show()
