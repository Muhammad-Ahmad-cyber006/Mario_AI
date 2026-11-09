import os
import sys

HERE=os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from dqn import DQN
from env import MarioEnv

try:
    best=DQN(MarioEnv(), MarioEnv(), resume='resume' in sys.argv).train()
    print(f'Done. Best score {best:.0f} saved in model.npz')
except KeyboardInterrupt:
    print('Stopped. Best model so far is saved in model.npz')
