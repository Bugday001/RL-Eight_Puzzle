# 八数码问题-强化学习
识别手写八数码问题，并使用强化学习求解。(Dueling DQN)

## 使用方法
执行`Get_loction.py`。出现检测界面，在检测界面按`s`即可开始求解八数码。

默认使用img中的图像识别。
若使用摄像头需将`is_video=0`改为`1`。

可以修改`mo_DQN_net.py`中的`difficulty_steps`来修改问题难度

## 文件
- `EightPuzzleEnv.py`八数码问题环境。
- `generator.py`八数码问题生成器。
- `Get_loction.py`程序入口。
- `mo_DQN_net.py`DQN程序。
- `move_render.py`求解可视化，移动数字。
- `number_recognition.py`数字识别
- `ReversePairs.py`判断是否有解。
- 其余为数字识别训练及测试用文件。

~~move_render的问题与图片位置太复杂，建议创建一个class记录每
个数字和其图片位置的对应关系，每次移动就改变~~

~~基础流程已经打通，但是还有很多可以优化的地方，比方说当截取图片太小时，
移动会出界，那么就需要在外面套一层边框(同时要修改图片原始位置)。~~