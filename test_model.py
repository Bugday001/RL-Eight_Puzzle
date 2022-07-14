from mo_DQN_net import *
from move_render import NumImg
import cv2


def play_test(problem_list, problem_loc, img=cv2.imread("./img/test1.png")):
    """
    :param problem_list: 以一维np array保存八数码问题 如 [4,1,2,7,6,5,0,8,3]。
    :param problem_loc:  图片各数字位置
    :param img: 图片
    :return:
    """
    # img show
    cv2.namedWindow('2', 0)
    cv2.resizeWindow('2', 640, 480)  # 自己设定窗口图片的大小
    agent = DQN()
    agent.load_model(111)
    # problem_list, problem_loc = minor_problem(loc, num_list)
    img_window = NumImg(img, problem_list, problem_loc)
    # s = env.reset3(difficulty_steps)
    s = env.reset2(problem_list.reshape(3, -1))
    s = s.flatten()  # 拉平
    while True:
        env.img_render(img_window)
        a = agent.predict(s)
        # take action
        s_, r, done, info = env.step(a)
        s_ = s_.flatten()  # 拉平
        # 存记忆, state, action, reward, next_state
        if done:
            s = s_
            env.img_render(img_window)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        s = s_


def play_without_img():
    agent = DQN()
    agent.load_model(111)
    s = env.reset3(difficulty_steps)
    s = s.flatten()  # 拉平
    while True:
        env.render()
        a = agent.predict(s)
        # take action
        s_, r, done, info = env.step(a)
        s_ = s_.flatten()  # 拉平
        # 存记忆, state, action, reward, next_state
        if done:
            s = s_
            env.render()
            break
        s = s_


if __name__ == "__main__":
    play_without_img()
