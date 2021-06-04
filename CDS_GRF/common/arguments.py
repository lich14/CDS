# -*- coding: utf-8 -*-

import argparse


def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--GPU', type=str, default="cuda:1", help='bool')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--last_action', type=bool, default=True,
                        help='whether to use the last action to choose action')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor')
    parser.add_argument('--optimizer', type=str,
                        default="RMS", help='the optimizer')
    parser.add_argument('--env', type=str, default='4_vs_2_full')

    parser.add_argument('--evaluate_epoch', type=int, default=20,
                        help='the number of the epoch to evaluate the agent')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result',
                        help='the result directory of the policy')

    parser.add_argument('--learn', type=bool, default=True,
                        help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='whether to use the GPU')
    parser.add_argument('--double_q', action='store_false', default=True)
    parser.add_argument('--dense_reward', action='store_true', default=False)
    parser.add_argument('--softmax_sample', action='store_true', default=False)

    # QPLEX
    parser.add_argument('--QPLEX_mixer', type=str,
                        default="dmaq_qatten", help='the difficulty of the game')
    parser.add_argument('--n_head', type=int, default=6)
    parser.add_argument('--num_kernel', type=int, default=6)
    parser.add_argument('--mixing_embed_dim', type=int, default=32)
    parser.add_argument('--hypernet_embed', type=int, default=64)

    parser.add_argument('--adv_hypernet_layers', type=int, default=1)
    parser.add_argument('--adv_hypernet_embed', type=int, default=64)

    parser.add_argument('--attend_reg_coef', type=float, default=0.001)
    parser.add_argument('--weighted_head', action='store_true', default=False)
    parser.add_argument('--nonlinear', action='store_true', default=False)
    parser.add_argument('--mask_dead', action='store_true', default=False)
    parser.add_argument('--state_bias', action='store_false', default=True)
    parser.add_argument('--is_minus_one', action='store_false', default=True)
    parser.add_argument('--burn_in_period', type=int, default=10)

    # Ours+QPLEX
    parser.add_argument('--beta1', type=float, default=.5)
    parser.add_argument('--beta2', type=float, default=2.)
    parser.add_argument('--beta', type=float, default=.05)

    parser.add_argument('--epsilon_anneal_time', type=int, default=500000)
    parser.add_argument('--train_steps', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--start_anneal_time', type=int, default=50000000)
    parser.add_argument('--anneal_rate', type=float, default=.3)
    parser.add_argument('--anneal_type', type=str, default='linear')
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--norm_weight', type=float, default=.1)
    parser.add_argument('--reuse_network', type=bool, default=False,
                        help='whether to use one network for all agents')

    # useless
    parser.add_argument('--noise_dim', type=int, default=1)

    args = parser.parse_args()
    return args


# arguments of vnd、 qmix、 qtran
def get_mixer_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.qtran_hidden_dim = 64
    args.lr = 5e-4

    args.epsilon_start = 1
    args.epsilon_finish = 0.05

    if args.QPLEX_mixer == "dmaq":
        args.weighted_head = True

    if args.env == '3_vs_2':
        args.alpha = 0.1
        args.epsilon_anneal_time = 50000
    elif args.env == '4_vs_3':
        args.alpha = 0.8
    elif args.env == '3_vs_2_full':
        args.alpha = 0.2

    args.n_epoch = 200000
    args.n_episodes = 1
    args.evaluate_cycle = 200

    args.buffer_size = int(5e3)
    args.save_cycle = 1000
    args.optim_alpha = 0.99
    args.optim_eps = 0.00001
    args.target_update_cycle = 200
    args.grad_norm_clip = 10
    args.num_kernel = args.n_head
    return args
