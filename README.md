# atari breakout game
> `breakout` `Reinforcement Learning` `DQN` `pytorch`

利用pytorch實作DQN在atari的breakout遊戲。

# Environment
## 1.套件安裝
- ``!pip install atari-py==0.2.6``
- ``!pip install gym==0.26.0``

# Intro
1.遊戲簡介

類似乒乓球的遊戲，玩家在遊戲中擁有五條生命，遊戲過程中須控制球拍擊球，將球反彈至上方擊打磚塊獲得積分，擊打到越上層的磚塊可以獲得越高的分數。若玩家未成功將球反彈而使其觸碰畫面下方邊界，則會失去一條生命，目標是盡可能獲得高分。

2.action space...
- action: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
- observation space: Box(0, 255, (210, 160, 3), uint8)
- import: gym.make("ALE/Breakout-v5")

# Implementation
1.Wrapper

2.Qnet

3.DQN
  deep Q-learning with experience replay

  q-vaule q-target

4.main loop

# Hyperparameters

# Experiments
