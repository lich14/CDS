# -*- coding: utf-8 -*-
import gym
import torch
import numpy as np
from env.wrapper_grf_3vs1 import make_football_env_3vs1
from env.wrapper_grf_3_vs_2_full import make_football_env_3vs2_full
from env.wrapper_grf_4_vs_2_full import make_football_env_4vs2_full
from env.wrapper_grf_4vs2 import make_football_env_4vs2
from env.wrapper_grf_4vs3 import make_football_env_4vs3
from env.wrapper_grf_4vs5 import make_football_env_4vs5
from env.wrapper_grf_11vs4 import make_football_env_11vs4
from env.wrapper_grf_5vs5 import make_football_env_5vs5
from runner import Runner
from common.arguments import get_common_args, get_mixer_args


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)


def get_env(args):
    if 'pacmen' in args.env:
        env = gym.make(args.env)
        env.seed(args.seed)

    elif args.env == '3_vs_2':
        # google research football
        env = make_football_env_3vs1(dense_reward=False)

    elif args.env == '3_vs_2_full':
        # google research football
        env = make_football_env_3vs2_full(dense_reward=False)

    elif args.env == '4_vs_2_full':
        # google research football
        env = make_football_env_4vs2_full(dense_reward=False)

    elif args.env == '4_vs_2':
        # google research football
        env = make_football_env_4vs2(dense_reward=False)

    elif args.env == '4_vs_3':
        # google research football
        env = make_football_env_4vs3(dense_reward=False)

    elif args.env == '4_vs_5':
        # google research football
        env = make_football_env_4vs5(dense_reward=False)

    elif args.env == '11_vs_4':
        # google research football
        env = make_football_env_11vs4(dense_reward=False)

    elif args.env == '5_vs_5':
        # google research football
        env = make_football_env_5vs5(dense_reward=False)

    return env


if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)
    import random
    args.seed = random.randint(0, 1000000)
    setup_seed(args.seed)
    env = get_env(args)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.reuse_network = False

    runner = Runner(env, args)
    runner.run(0)
    env.close()
