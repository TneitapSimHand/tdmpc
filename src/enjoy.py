import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MUJOCO_GL'] = 'egl' # comment on windows, according to https://github.com/mkocabas/VIBE/issues/101
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import Episode, ReplayBuffer
import logger
import cv2
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def enjoy(cfg):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
    agent.load("logs/dog-run/state/default/1/models/model.pt")

    for step in range(0, 1000):
        # Collect trajectory
        obs = env.reset()
        episode = Episode(cfg, obs)
        while not episode.done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)
            img_rgb = env.render()
            cv2.imshow("img_rgb", img_rgb[:, :, ::-1])
            cv2.waitKey(1)



if __name__ == '__main__':
    enjoy(parse_cfg(Path().cwd() / __CONFIG__))