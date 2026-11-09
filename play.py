import os
import sys

HERE=os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import numpy as np
import pygame as pg

from Const import FPS
from dqn import load_model
from env import MarioEnv, run_episode

net=load_model()
env=MarioEnv(render=True)
clock=pg.time.Clock()


font=pg.font.Font(None, 24)


# shows which buttons the AI presses and what Mario is doing (up/down, forward/back)
def overlay(e):
    player=e.map.get_player()
    x=10
    for name, pressed in (('LEFT', e.core.keyL), ('RIGHT', e.core.keyR), ('JUMP', e.core.keyU), ('RUN', e.core.keyShift)):
        label=font.render(name, True, (0, 0, 0) if pressed else (200, 200, 200), (255, 220, 0) if pressed else (60, 60, 60))
        e.screen.blit(label, (x, 60))
        x+=label.get_width()+8

    vertical='on ground' if player.on_ground else 'going UP' if player.y_vel<0 else 'going DOWN'
    horizontal='moving BACK' if player.x_vel<0 else 'moving FORWARD' if player.x_vel>0 else 'standing still'
    text=f'{vertical} | {horizontal} | x={player.rect.x} furthest={e.max_x} score={player.score}'
    e.screen.blit(font.render(text, True, (255, 255, 0), (0, 0, 0)), (10, 86))


def on_frame(e):
    for ev in pg.event.get():
        if ev.type==pg.QUIT or (ev.type==pg.KEYDOWN and ev.key==pg.K_ESCAPE):
            sys.exit()
    e.draw()
    overlay(e)
    pg.display.flip()
    clock.tick(FPS)


while True:
    info=run_episode(env, lambda obs: int(np.argmax(net.forward(obs))), on_frame)
    print(f'{"WON" if info["won"] else "died" if info["died"] else "timeout"} score={info["score"]} distance={info["distance"]}', flush=True)
