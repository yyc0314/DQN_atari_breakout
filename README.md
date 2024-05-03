# atari breakout game
> `breakout` `Reinforcement Learning` `DQN` `pytorch`

利用pytorch實作DQN在atari的breakout遊戲。

# Environment
1.環境(Initialization)
```python
!wget http://www.atarimania.com/roms/Roms.rar
!mkdir /content/ROM/
!unrar e /content/Roms.rar /content/ROM/
!python -m atari_py.import_roms /content/ROM/
!pip install atari-py==0.2.6
!pip install gym==0.21.0
!pip install gym[atari]
```
2.套件安裝

# Intro
1.遊戲簡介

2.action space...

# Implementation
1.Wrapper

2.Qnet

3.DQN
  deep Q-learning with experience replay

  q-vaule q-target

4.main loop

# Hyperparameters

# Experiments
