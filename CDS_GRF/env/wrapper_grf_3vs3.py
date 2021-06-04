import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
import matplotlib.pyplot as plt


class GoogleFootballMultiAgentEnv(object):
    """An wrapper for GFootball to make it compatible with our codebase."""

    def __init__(self, dense_reward, dump_freq, render=False):
        self.nagents = 2
        self.time_limit = 500
        self.time_step = 0
        self.obs_dim = 28  # for 3 vs 3
        self.dense_reward = dense_reward  # select whether to use dense reward

        # Instantiate environment

        self.env = football_env.create_environment(
            env_name='3_vs_3',
            stacked=False,
            representation="simple115",
            rewards='scoring',
            logdir='football_dumps',
            render=render,
            write_video=False,
            dump_frequency=dump_freq,
            number_of_left_players_agent_controls=self.nagents,
            number_of_right_players_agent_controls=0,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.nagents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.nagents)
        ]
        self.reward_record = [0, 0]

        self.game_mode = 0
        self.old_game_mode = 0
        self.goal_owned_team = -1

    def get_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        # Add some features to discribe offside
        '''pos_1st_ours = full_obs['left_team'][np.argmax(
            full_obs['left_team'][1:, 0])]

        _, col = np.unravel_index(
            np.argsort(full_obs['right_team'][:, 0].reshape(1, -1).ravel()), full_obs['right_team'][:, 0].reshape(1, -1).shape)
        pos_2nd_opponent = full_obs['right_team'][col[-2]]

        pos_1st_opponent = full_obs['right_team'][np.argmin(
            full_obs['right_team'][1:, 0])]

        _, col = np.unravel_index(
            np.argsort(full_obs['left_team'][:, 0].reshape(1, -1).ravel()), full_obs['left_team'][:, 0].reshape(1, -1).shape)
        pos_2nd_ours = full_obs['left_team'][col[2]]

        offside_ours = max(pos_2nd_opponent[0], 0) - pos_1st_ours[0]
        offside_opponent = pos_1st_opponent[0] - min(pos_2nd_ours[0], 0)'''
        self.old_game_mode = self.game_mode
        self.game_mode = full_obs['game_mode']
        if full_obs['ball_owned_team'] != -1:
            self.goal_owned_team = full_obs['ball_owned_team']

        if index == -1:
            # global state, absolute position
            # cannot control our goal keeper
            simple_obs.append(full_obs['left_team']
                              [-self.nagents:].reshape(-1))
            simple_obs.append(
                full_obs['left_team_direction'][-self.nagents:].reshape(-1))

            simple_obs.append(full_obs['right_team'].reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

            '''simple_obs.append(pos_1st_ours)
            simple_obs.append(pos_1st_opponent)

            simple_obs.append(pos_2nd_ours)
            simple_obs.append(pos_2nd_opponent)

            simple_obs.append(np.array([offside_ours]))
            simple_obs.append(np.array([offside_opponent]))'''
            simple_obs.append(np.array([self.game_mode]))
            simple_obs.append(np.array([full_obs['ball_owned_team']]))

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.nagents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.nagents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.nagents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.nagents:], index, axis=0).reshape(-1))

            simple_obs.append(
                (full_obs['right_team'] - ego_position).reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

            '''offside_ego = max(
                pos_2nd_opponent[0], 0) - ego_position[0]  
            simple_obs.append(pos_1st_ours - ego_position)
            simple_obs.append(pos_1st_opponent - ego_position)

            simple_obs.append(pos_2nd_ours - ego_position)
            simple_obs.append(pos_2nd_opponent - ego_position)

            simple_obs.append(np.array([offside_ours]))
            simple_obs.append(np.array([offside_ego]))
            simple_obs.append(np.array([offside_opponent]))'''
            simple_obs.append(np.array([self.game_mode]))
            simple_obs.append(np.array([full_obs['ball_owned_team']]))

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_global_state(self):
        return self.get_obs(-1)

    def check_if_our_out(self):

        if self.old_game_mode != self.game_mode and self.game_mode != 0:
            # -1 = ball not owned, 0 = left team, 1 = right team.
            if self.goal_owned_team == 0:
                return -1
            elif self.goal_owned_team == 1:
                return 1

        return 0

    def reset(self):
        self.time_step = 0
        self.env.reset()
        obs = np.array([self.get_obs(i) for i in range(self.nagents)])
        self.reward_record = [0, 0]

        self.game_mode = 0
        self.old_game_mode = 0
        self.goal_owned_team = -1

        return obs, self.get_global_state()

    def step(self, actions):

        self.time_step += 1
        _, original_rewards, done, infos = self.env.step(
            actions + 1)  # cancel not move action
        obs = np.array([self.get_obs(i) for i in range(self.nagents)])

        if self.time_step >= self.time_limit:
            done = True

        if sum(original_rewards) < 0:
            self.reward_record[1] += 1
        if sum(original_rewards) > 0:
            self.reward_record[0] += 1

        infos['reward_record'] = self.reward_record
        if sum(original_rewards) == 0:
            return obs, self.get_global_state(), self.check_if_our_out(), done, infos

        if sum(original_rewards) < 0:
            return obs, self.get_global_state(), -100, done, infos

        return obs, self.get_global_state(), 100, done, infos

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def get_env_info(self):
        output_dict = {}
        output_dict['n_actions'] = self.action_space[0].n - 1
        output_dict['obs_shape'] = self.obs_dim
        output_dict['n_agents'] = 2
        output_dict['state_shape'] = self.obs_dim - 1
        output_dict['episode_limit'] = self.time_limit

        return output_dict


# def make_football_env(seed_dir, dump_freq=1000, representation='extracted', render=False):
def make_football_env_3vs3(dense_reward=False, dump_freq=1000):
    '''
    Creates a env object. This can be used similar to a gym
    environment by calling env.reset() and env.step().

    Some useful env properties:
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .nagents            :   Returns the number of Agents
    '''
    return GoogleFootballMultiAgentEnv(dense_reward, dump_freq)


if __name__ == "__main__":
    env = make_football_env_3vs3()
    a, b = env.reset()
    print(a.shape, b.shape)
