# atari breakout game
> `breakout` `Reinforcement Learning` `DQN` `pytorch`

利用pytorch實作DQN在atari的breakout遊戲。

# Environment
## 1.套件安裝
- ``!pip install atari-py==0.2.6``
- ``!pip install gymnasium==0.26.0``
- ``!pip install gymnasium[atari]``

# Intro
## 1.遊戲簡介

類似乒乓球的遊戲，玩家在遊戲中擁有五條生命，遊戲過程中須控制球拍擊球，將球反彈至上方擊打磚塊獲得積分，擊打到越上層的磚塊可以獲得越高的分數。若玩家未成功將球反彈而使其觸碰畫面下方邊界，則會失去一條生命，目標是盡可能獲得高分。

## 2.game information
- action: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
- observation space: Box(0, 255, (210, 160, 3), uint8)
- import: evn=gym.make("ALE/Breakout-v5")

## 3.RL運作原理
```python
state=env.reset() #初始化
while True:
  action=choose_action(state) #RL演算法
  state_next,reward,ter,tru,info=env.step(action)
  #ter->terminated 遊戲成功或失敗，tru->truncated 是否超過遊戲限制(ex:超時、超步數)，info->除了state以外的其他資訊

  env.render() #畫面渲染
  state=state_next.copy() #更新狀態
  if ter or tru:
    break
```

# Implementation
## 1.Wrapper
- gymnasium.Wrapper
  - env.reset():初始化
  - env.step(action):選擇一個動作，環境回傳**state_next,reward,ter,tru,info**
- 遊戲畫面預處理
  - Resize(調整大小):(84,84)
  - Grayscale Conversion(灰階轉換):將RGB值除以255轉灰階
  - Frame Stacking(幀疊加):一次疊加4幀畫面

2.Qnet

3.DQN
  deep Q-learning with experience replay

  q-vaule q-target

4.main loop

# Hyperparameters

# Experiments
