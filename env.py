import os

import numpy as np

from Const import FPS, MAX_FASTMOVE_SPEED, MAX_FALL_SPEED

MODEL='model.npz'
FRAME_SKIP=4
MAX_STEPS=3000
STALL_LIMIT=80
BEHIND, AHEAD, ABOVE, BELOW=4, 9, 5, 2
COLS=BEHIND+AHEAD+1
ROWS=ABOVE+BELOW+1
OBS_SIZE=COLS*ROWS+6

PROGRESS_SCALE=0.01
SCORE_SCALE=0.02
DEATH_PENALTY=10.0
STALL_PENALTY=5.0
WIN_BONUS=10.0

# right, left, jump, run
BUTTONS=(
    (False, False, False, False),
    (True,  False, False, False),
    (True,  False, True,  False),
    (True,  False, False, True),
    (True,  False, True,  True),
    (False, False, True,  False),
    (False, True,  False, False),
    (False, True,  True,  False),
    (False, True,  False, True),
    (False, True,  True,  True),
)
N_ACTIONS=len(BUTTONS)

# the game's kill-streak timer uses the real clock, so headless runs get a simulated one
clock={'ms':0}


def observe(game_map):
    player=game_map.get_player()
    cx=player.rect.centerx//32
    cy=player.rect.centery//32
    width, height=game_map.mapSize
    grid=np.zeros((ROWS, COLS), dtype=np.float32)

    for r in range(ROWS):
        for c in range(COLS):
            tx=cx-BEHIND+c
            ty=cy-ABOVE+r

            if tx<0:   # a negative index would wrap around to the end of the level
                grid[r, c]=1.0
            elif tx<width and 0<=ty<height:
                tile=game_map.map[tx][ty]

                if tile!=0 and tile.type!='BGObject':
                    unused_question=getattr(tile, 'typeID', None)==22 and not tile.isActivated
                    grid[r, c]=0.25 if unused_question else 1.0

    for mob in game_map.get_mobs():
        name=type(mob).__name__

        if name in ('Goombas', 'Koopa') and mob.state!=-1:
            value=-1.0
        elif name in ('Mushroom', 'Flower'):
            value=0.5
        else:
            continue

        c=mob.rect.centerx//32-cx+BEHIND
        r=mob.rect.centery//32-cy+ABOVE
        if 0<=r<ROWS and 0<=c<COLS:
            grid[r, c]=value

    extra=[player.x_vel/MAX_FASTMOVE_SPEED, player.y_vel/MAX_FALL_SPEED, float(player.on_ground),
           player.powerLVL/2.0, (player.rect.centerx%32)/32, (player.rect.centery%32)/32]
    return np.concatenate([grid.ravel(), np.array(extra, dtype=np.float32)])


class NullSound:
    def play(self, *args, **kwargs):
        pass

    stop=start_fast_music=play


class HeadlessCore:
    def __init__(self, screen):
        self.screen=screen
        self.map=None
        self.sound=NullSound()
        self.keyR=self.keyL=self.keyU=self.keyD=self.keyShift=False

    def get_map(self):
        return self.map

    def get_sound(self):
        return self.sound


class MarioEnv:
    def __init__(self, render=False):
        if not render:
            os.environ['SDL_VIDEODRIVER']='dummy'   # before pygame starts

        import pygame as pg
        from Const import WINDOW_W, WINDOW_H
        from Map import Map

        pg.time.get_ticks=lambda: clock['ms']
        pg.display.init()
        pg.font.init()
        self.screen=pg.display.set_mode((WINDOW_W, WINDOW_H))

        self.core=HeadlessCore(self.screen)
        self.map=Map('1-1')
        self.core.map=self.map
        self.done=True

        # flat ground with free air above: not a pit, pipe or stair
        tiles=self.map.map
        solid=lambda tx, ty: tiles[tx][ty]!=0 and tiles[tx][ty].type!='BGObject'
        last=int(self.map.flag.pillar_rect.x//32)-3
        self.start_columns=[tx for tx in range(2, last)
                            if solid(tx, 12) and solid(tx+1, 12)
                            and not any(solid(tx, ty) or solid(tx+1, ty) for ty in (9, 10, 11))]

    # start_x=None starts at the real start, otherwise Mario is teleported there
    def reset(self, start_x=None):
        self.map.reset(True)
        self.map.score_for_killing_mob=100   # the game never resets the kill streak
        player=self.map.get_player()

        if start_x is not None:
            player.rect.x=player.pos_x=int(start_x)

            # spawn the levels enemies, then clear those within 6 tiles so Mario survives frame 1
            self.map.try_spawn_mobs(self.core)
            self.map.try_spawn_mobs(self.core)
            self.map.mobs=[m for m in self.map.mobs if abs(m.rect.centerx-player.rect.centerx)>192]

        self.core.keyR=self.core.keyL=self.core.keyU=self.core.keyD=self.core.keyShift=False
        self.blocks=[o for o in self.map.obj if getattr(o, 'typeID', 0) in (22, 23)]
        self.start_x=self.max_x=player.rect.x
        self.steps=0
        self.since_progress=0
        self.done=False
        return observe(self.map)

    def step(self, action, on_frame=None):
        if self.done:
            raise RuntimeError('episode is over, call reset()')

        self.core.keyR, self.core.keyL, self.core.keyU, self.core.keyShift=BUTTONS[action]
        player=self.map.get_player()
        prev_x=player.rect.x
        prev_score=player.score

        for _ in range(FRAME_SKIP):
            clock['ms']+=1000//FPS
            self.map.update(self.core)

            # the game moves a hit blocks rect only inside render(); do the same when not drawing
            if on_frame is None:
                for block in self.blocks:
                    if block.shaking:
                        block.shake()

            if player.rect.x<0:   # invisible wall at the left edge
                player.rect.x=player.pos_x=0
                player.x_vel=max(player.x_vel, 0)
            if on_frame is not None:
                on_frame(self)
            if self.map.in_event:
                break

        # in_event is set only by player_death() (type 0) or player_win() (type 1)
        died=self.map.in_event and self.map.get_event().type==0
        won=self.map.in_event and self.map.get_event().type==1

        x=player.rect.x
        self.steps+=1
        self.since_progress=0 if x>self.max_x or player.score>prev_score else self.since_progress+1   # scoring counts as progress
        self.max_x=max(self.max_x, x)

        stalled=not died and not won and self.since_progress>=STALL_LIMIT
        terminated=died or won or stalled
        truncated=not terminated and self.steps>=MAX_STEPS
        self.done=terminated or truncated

        reward=(x-prev_x)*PROGRESS_SCALE+(player.score-prev_score)*SCORE_SCALE
        reward+=WIN_BONUS*won-DEATH_PENALTY*died-STALL_PENALTY*stalled

        info={'distance':self.max_x-self.start_x, 'steps':self.steps, 'died':died, 'won':won, 'score':player.score}
        return observe(self.map), reward, terminated, truncated, info

    def draw(self):
        self.map.render(self.core)


def episode_score(info):
    return float(info['score'])


def run_episode(env, act, on_frame=None):
    obs=env.reset()

    while True:
        obs, _, terminated, truncated, info=env.step(act(obs), on_frame=on_frame)
        if terminated or truncated:
            return info
