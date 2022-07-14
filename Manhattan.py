import numpy as np

target = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

state = np.array([[2,8,3],[1,0,4],[7,6,5]])


def distance():
    dist = 0
    for num in range(9):
        t = np.array(np.where(target == num))
        st = np.array(np.where(state == num))
        dist += np.sum(abs(t-st))
    print(dist)


if __name__ == "__main__":
    distance()