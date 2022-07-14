import numpy


def puzzle_generator(deq):
    # target = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
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
    self.state = nextState