import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.base_net import MLP, RNN
from network.predict_net import Predict_Network1, Predict_Network1_combine
from network.QPLEX.dmaq_general import DMAQer
from network.QPLEX.dmaq_qatten import DMAQ_QattenMixer
from torch.optim import RMSprop
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class DMAQ_qattenLearner_SDQ_intrinsic:

    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.args = args

        if args.QPLEX_mixer == "dmaq":
            self.eval_mix_net = DMAQer(args)
            self.target_mix_net = DMAQer(args)
        elif args.QPLEX_mixer == 'dmaq_qatten':
            self.eval_mix_net = DMAQ_QattenMixer(args)
            self.target_mix_net = DMAQ_QattenMixer(args)
        else:
            raise ValueError(
                "Mixer {} not recognised.".format(args.QPLEX_mixer))

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)

        self.eval_mlp = nn.ModuleList([MLP(args)
                                       for _ in range(args.n_agents)])
        self.target_mlp = nn.ModuleList(
            [MLP(args) for _ in range(args.n_agents)])

        self.eval_predict_withoutid = Predict_Network1(
            args.rnn_hidden_dim + args.obs_shape + args.n_actions, 128, args.obs_shape, False)
        self.target_predict_withoutid = Predict_Network1(
            args.rnn_hidden_dim + args.obs_shape + args.n_actions, 128, args.obs_shape, False)

        self.eval_predict_withid = Predict_Network1_combine(args.rnn_hidden_dim + args.obs_shape + args.n_actions + args.n_agents, 128,
                                                            args.obs_shape, args.n_agents, False)
        self.target_predict_withid = Predict_Network1_combine(args.rnn_hidden_dim + args.obs_shape + args.n_actions + args.n_agents, 128,
                                                              args.obs_shape, args.n_agents, False)

        if self.args.cuda:
            self.eval_rnn.to(torch.device(self.args.GPU))
            self.target_rnn.to(torch.device(self.args.GPU))
            self.eval_mix_net.to(torch.device(self.args.GPU))
            self.target_mix_net.to(torch.device(self.args.GPU))

            self.eval_mlp.to(torch.device(self.args.GPU))
            self.target_mlp.to(torch.device(self.args.GPU))

            self.eval_predict_withid.to(torch.device(self.args.GPU))
            self.target_predict_withid.to(torch.device(self.args.GPU))

            self.eval_predict_withoutid.to(torch.device(self.args.GPU))
            self.target_predict_withoutid.to(torch.device(self.args.GPU))

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
        self.target_predict_withid.load_state_dict(
            self.eval_predict_withid.state_dict())
        self.target_predict_withoutid.load_state_dict(
            self.eval_predict_withoutid.state_dict())

        self.eval_parameters = list(self.eval_mix_net.parameters(
        )) + list(self.eval_rnn.parameters()) + list(self.eval_mlp.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(
                self.eval_parameters, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.model_dir = f'{args.model_dir}/{args.env}/seed_{args.seed}'

        self.eval_hidden = None
        self.target_hidden = None

    def learn(self, batch, max_episode_len, train_step, t_env, epsilon=None):

        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
            batch['r'],  batch['avail_u'], batch['avail_u_next'],\
            batch['terminated']
        mask = 1 - batch["padded"].float()
        obs, obs_next = batch['o'], batch['o_next']

        q_evals, q_targets, q_evals_local, q_evals_last, intrinsic_rewards = self.get_q_values(
            batch, max_episode_len)
        if t_env > self.args.start_anneal_time:
            if self.args.anneal_type == 'linear':
                intrinsic_rewards = max(1 - self.args.anneal_rate * (
                    t_env - self.args.start_anneal_time) / 1000000, 0) * intrinsic_rewards
            elif self.args.anneal_type == 'exp':
                exp_scaling = (-1) * (1 / self.args.anneal_rate) / np.log(0.01)
                TTT = (t_env - self.args.start_anneal_time) / 1000000
                intrinsic_rewards = intrinsic_rewards * \
                    min(1, max(0.01, np.exp(-TTT / exp_scaling)))

        mac_out = q_evals.clone().detach()

        if self.args.cuda:
            obs = obs.to(torch.device(self.args.GPU))
            obs_next = obs.to(torch.device(self.args.GPU))
            s = s.to(torch.device(self.args.GPU))
            u = u.to(torch.device(self.args.GPU))
            r = r.to(torch.device(self.args.GPU))
            s_next = s_next.to(torch.device(self.args.GPU))
            terminated = terminated.to(torch.device(self.args.GPU))
            mask = mask.to(torch.device(self.args.GPU))

        max_action_qvals, _ = q_evals.max(dim=3)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        curr_actions_onehot = torch.zeros(
            u.squeeze(3).shape + (self.n_actions,))
        if self.args.cuda:
            curr_actions_onehot = curr_actions_onehot.to(
                torch.device(self.args.GPU))

        curr_actions_onehot = curr_actions_onehot.scatter_(3, u, 1)

        with torch.no_grad():

            q_targets[avail_u_next == 0.0] = -9999999

            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out[avail_u == 0] = -9999999
                cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
                target_last_max_actions = q_evals_last.unsqueeze(
                    1).max(dim=3, keepdim=True)[1]
                double_max_actions = torch.cat(
                    [cur_max_actions, target_last_max_actions], dim=1)
                target_max_qvals = q_targets.max(dim=3)[0]
                q_targets = torch.gather(
                    q_targets, 3, double_max_actions).squeeze(3)

                cur_max_actions_onehot = torch.zeros(
                    double_max_actions.squeeze(3).shape + (self.n_actions,))
                if self.args.cuda:
                    cur_max_actions_onehot = cur_max_actions_onehot.to(
                        torch.device(self.args.GPU))
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(
                    3, double_max_actions, 1)

            else:
                q_targets = q_targets.max(dim=3)[0]
                target_max_qvals = q_targets.max(dim=3)[0]

        if self.args.QPLEX_mixer == "dmaq_qatten":
            ans_chosen, q_attend_regs, _ = self.eval_mix_net(
                q_evals, s, obs, is_v=True)
            ans_adv, _, _ = self.eval_mix_net(
                q_evals, s, obs, actions=curr_actions_onehot, max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv
        else:
            ans_chosen = self.eval_mix_net(q_evals, s, is_v=True)
            ans_adv = self.eval_mix_net(
                q_evals, s, actions=curr_actions_onehot, max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv

        with torch.no_grad():
            if self.args.double_q:
                if self.args.QPLEX_mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mix_net(
                        q_targets, s_next, obs_next, is_v=True)
                    target_adv, _, _ = self.target_mix_net(
                        q_targets, s_next, obs_next, actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mix_net(
                        q_targets, s_next, is_v=True)
                    target_adv = self.target_mix_net(
                        q_targets, s_next, actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mix_net(
                    target_max_qvals, s_next, is_v=True)

            # Calculate 1-step Q-Learning targets
            targets = r + self.args.beta * \
                intrinsic_rewards.mean(
                    dim=1) + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        update_prior = (masked_td_error**2).squeeze().sum(dim=-1,
                                                          keepdim=True) / mask.squeeze().sum(dim=-1, keepdim=True)

        # Normal L2 loss, take mean over actual data
        if self.args.QPLEX_mixer == "dmaq_qatten":
            loss = (masked_td_error**2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error**2).sum() / mask.sum()

        norm_loss = F.l1_loss(q_evals_local, target=torch.zeros_like(
            q_evals_local), size_average=True)
        loss += self.args.norm_weight * norm_loss

        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict())

            self.target_predict_withid.load_state_dict(
                self.eval_predict_withid.state_dict())
            self.target_predict_withoutid.load_state_dict(
                self.eval_predict_withoutid.state_dict())

        return update_prior.squeeze().detach()

    def _get_inputs_matrix(self, batch):
        obs, obs_next = batch['o'], batch['o_next']

        obs_clone = obs.clone()
        obs_next_clone = obs_next.clone()

        if self.args.last_action:
            u_onehot = batch['u_onehot']
            u_onehot_f = torch.zeros_like(u_onehot)
            u_onehot_f[:, 1:, :, :] = u_onehot[:, :-1, :, :]

            obs = torch.cat([obs, u_onehot_f], dim=-1)
            obs_next = torch.cat([obs_next, u_onehot], dim=-1)

        add_id = torch.eye(self.args.n_agents).type_as(obs).expand(
            [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])

        if self.args.reuse_network:
            obs = torch.cat([obs, add_id], dim=-1)
            obs_next = torch.cat([obs_next, add_id], dim=-1)

        return obs, obs_next, obs_clone, obs_next_clone, add_id

    def get_q_values(self, batch, max_episode_len):

        inputs, inputs_next, obs, obs_next, add_id = self._get_inputs_matrix(
            batch)
        inputs = torch.cat([inputs, inputs_next[:, -1].unsqueeze(1)], dim=1)
        inputs_shape = inputs.shape
        mask = 1 - batch["padded"].float()

        if self.args.cuda:
            inputs = inputs.to(torch.device(self.args.GPU))
            inputs_next = inputs_next.to(torch.device(self.args.GPU))
            obs = obs.to(torch.device(self.args.GPU))
            obs_next = obs_next.to(torch.device(self.args.GPU))
            mask = mask.to(torch.device(self.args.GPU))

            self.eval_hidden = self.eval_hidden.to(torch.device(self.args.GPU))
            self.target_hidden = self.target_hidden.to(
                torch.device(self.args.GPU))

        u_onehot = batch['u_onehot']
        u_onehot = u_onehot.to(inputs.device).permute(0, 2, 1, 3)
        add_id = add_id.to(inputs.device).permute(0, 2, 1, 3)

        eval_h = self.eval_hidden.view(-1, self.args.rnn_hidden_dim)
        target_h = self.target_hidden.view(-1, self.args.rnn_hidden_dim)

        inputs = inputs.permute(0, 2, 1, 3)
        inputs_next = inputs_next.permute(0, 2, 1, 3)

        inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3])
        inputs_next = inputs_next.reshape(-1,
                                          inputs_next.shape[2], inputs_next.shape[3])

        q_eval_global, out_eval_h = self.eval_rnn(inputs, eval_h)
        q_target_global, out_target_h = self.target_rnn(inputs_next, target_h)

        q_eval_global = q_eval_global.reshape(inputs_shape[0], inputs_shape[2], q_eval_global.shape[-2],
                                              q_eval_global.shape[-1]).permute(0, 2, 1, 3)

        out_eval_h = out_eval_h.reshape(
            inputs_shape[0], inputs_shape[2], out_eval_h.shape[-2], out_eval_h.shape[-1]).permute(0, 2, 1, 3)

        q_target_global = q_target_global.reshape(inputs_shape[0], inputs_shape[2], q_target_global.shape[-2],
                                                  q_target_global.shape[-1]).permute(0, 2, 1, 3)
        out_target_h = out_target_h.reshape(inputs_shape[0], inputs_shape[2], out_target_h.shape[-2],
                                            out_target_h.shape[-1]).permute(0, 2, 1, 3)

        q_eval_local = torch.stack(
            [self.eval_mlp[id](out_eval_h[:, :, id].reshape(-1, out_eval_h.shape[-1]))
             for id in range(self.args.n_agents)],
            dim=1).reshape_as(q_eval_global)

        q_target_local = torch.stack(
            [self.target_mlp[id](out_target_h[:, :, id].reshape(-1, out_target_h.shape[-1]))
             for id in range(self.args.n_agents)],
            dim=1).reshape_as(q_target_global)

        q_eval = q_eval_global + q_eval_local
        q_target = q_target_global + q_target_local

        with torch.no_grad():
            mask = mask.unsqueeze(-2).expand(obs.shape[:-1] + mask.shape[-1:])
            mask = mask.permute(0, 2, 1, 3)
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
            mask = mask.reshape(-1, mask.shape[-1])

            obs_intrinsic = obs.clone().permute(0, 2, 1, 3)
            obs_intrinsic = obs_intrinsic.reshape(
                -1, obs_intrinsic.shape[-2], obs_intrinsic.shape[-1])

            eval_h_intrinsic = out_eval_h.clone().permute(0, 2, 1, 3)
            eval_h_intrinsic = eval_h_intrinsic.reshape(
                -1, eval_h_intrinsic.shape[-2], eval_h_intrinsic.shape[-1])

            h_cat = torch.cat([self.eval_hidden.reshape(-1, self.eval_hidden.shape[-1]
                                                        ).unsqueeze(1), eval_h_intrinsic[:, :-2]], dim=1)

            intrinsic_input_1 = torch.cat(
                [h_cat, obs_intrinsic, u_onehot.reshape(-1, u_onehot.shape[-2], u_onehot.shape[-1])], dim=-1)
            intrinsic_input_2 = torch.cat(
                [intrinsic_input_1, add_id.reshape(-1, add_id.shape[-2], add_id.shape[-1])], dim=-1)

            intrinsic_input_1 = intrinsic_input_1.reshape(
                -1, intrinsic_input_1.shape[-1])
            intrinsic_input_2 = intrinsic_input_2.reshape(
                -1, intrinsic_input_2.shape[-1])

            next_obs_intrinsic = obs_next.clone().permute(0, 2, 1, 3)
            next_obs_intrinsic = next_obs_intrinsic.reshape(
                -1, next_obs_intrinsic.shape[-2], next_obs_intrinsic.shape[-1])
            next_obs_intrinsic = next_obs_intrinsic.reshape(
                -1, next_obs_intrinsic.shape[-1])

            log_p_o = self.target_predict_withoutid.get_log_pi(
                intrinsic_input_1, next_obs_intrinsic)
            log_q_o = self.target_predict_withid.get_log_pi(
                intrinsic_input_2, next_obs_intrinsic, add_id.reshape([-1, add_id.shape[-1]]))

            mean_p = torch.softmax(q_eval[:, :-1], dim=-1).mean(dim=2)
            q_pi = torch.softmax(self.args.beta1 * q_eval[:, :-1], dim=-1)

            pi_diverge = torch.cat(
                [(q_pi[:, :, id] * torch.log(q_pi[:, :, id] / mean_p)
                  ).sum(dim=-1, keepdim=True) for id in range(self.n_agents)],
                dim=-1).permute(0, 2, 1).unsqueeze(-1)

            intrinsic_rewards = self.args.beta1 * log_q_o - log_p_o
            intrinsic_rewards = intrinsic_rewards.reshape(
                -1, obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])
            intrinsic_rewards = intrinsic_rewards.reshape(
                -1, obs.shape[2], obs_intrinsic.shape[1], intrinsic_rewards.shape[-1])
            intrinsic_rewards = intrinsic_rewards + self.args.beta2 * pi_diverge

        # update predict network
        add_id = add_id.reshape([-1, add_id.shape[-1]])
        for index in BatchSampler(SubsetRandomSampler(range(intrinsic_input_1.shape[0])), 256, False):
            self.eval_predict_withoutid.update(
                intrinsic_input_1[index], next_obs_intrinsic[index], mask[index])
            self.eval_predict_withid.update(
                intrinsic_input_2[index], next_obs_intrinsic[index], add_id[index], mask[index])

        return q_eval[:, :-1], q_target, q_eval_local[:, :-1], q_eval[:, -1], intrinsic_rewards.detach()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.eval_mlp.state_dict(), self.model_dir +
                   '/' + num + '_mlp_net_params.pkl')
        torch.save(self.eval_mix_net.state_dict(), self.model_dir +
                   '/' + num + '_qplex_mix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir +
                   '/' + num + '_rnn_net_params.pkl')

    def load_model(self, step):
        self.eval_mlp.load_state_dict(torch.load(
            f"{self.model_dir}/{step}_mlp_net_params.pkl", map_location='cpu'))
        self.eval_mix_net.load_state_dict(torch.load(
            f"{self.model_dir}/{step}_qplex_mix_net_params.pkl", map_location='cpu'))
        self.eval_rnn.load_state_dict(torch.load(
            f"{self.model_dir}/{step}_rnn_net_params.pkl", map_location='cpu'))

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            episode_num, self.n_agents, self.args.rnn_hidden_dim)
        self.target_hidden = torch.zeros(
            episode_num, self.n_agents, self.args.rnn_hidden_dim)
        if self.args.cuda:
            self.eval_hidden = self.eval_hidden.to(torch.device(self.args.GPU))
            self.target_hidden = self.target_hidden.to(
                torch.device(self.args.GPU))

        for i in range(episode_num):
            for j in range(self.n_agents):
                self.eval_hidden[i, j] = self.eval_rnn.init_hidden()
                self.target_hidden[i, j] = self.target_rnn.init_hidden()
