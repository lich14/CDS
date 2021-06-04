import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
import matplotlib.pyplot as plt


class GoogleFootballMultiAgentEnv(object):
    """An wrapper for GFootball to make it compatible with our codebase."""

    def __init__(self, dense_reward, dump_freq, render=False):
        self.nagents = 10
        self.time_limit = 400
        self.time_step = 0
        self.obs_dim = 62  # for 11 vs 4
        self.dense_reward = dense_reward  # select whether to use dense reward

        # Instantiate environment

        self.env = football_env.create_environment(
            env_name='11_vs_4_offence_stochastic',
            stacked=False,
            representation="simple115",
            rewards='scoring',
            logdir='football_dumps',
            render=render,
            write_video=True,
            dump_frequency=dump_freq,
            number_of_left_players_agent_controls=10,
            number_of_right_players_agent_controls=0,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        # Define some useful attributes

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.nagents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.nagents)
        ]

    '''def convert_simple115_to_simple37(self, simple115_vectors):
        assert simple115_vectors.shape == (self.nagents, 115) or simple115_vectors.shape == (1, 115)

        # In academy_3_vs_1_with_keeper, a bunch of players and game modes are irrelevant
        # so we remove the associated observations from the simple115 vector to a simple37 vector
        obs_to_remove = np.concatenate([
            np.arange(24, 88),  # coordinates and movement directions of absent players
            np.arange(101, 108),  # indices of active_player one-hot encoding corresponding to absent players
            np.arange(108, 115)  # indices of game_mode one-hot encoding (only one game_mode is used here)
        ])

        return np.delete(simple115_vectors, obs_to_remove, axis=1)'''

    def get_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

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

        simple_obs = np.concatenate(simple_obs)
        return simple_obs

    def get_global_state(self):
        return self.get_obs(-1)

    def reset(self):
        self.time_step = 0
        self.env.reset()
        obs = np.array([self.get_obs(i) for i in range(self.nagents)])

        return obs, self.get_global_state()

    def step(self, actions):

        self.time_step += 1
        _, original_rewards, done, infos = self.env.step(actions)
        rewards = list(original_rewards)
        obs = np.array([self.get_obs(i) for i in range(self.nagents)])

        if self.time_step >= self.time_limit:
            done = True

        if sum(rewards) <= 0:
            return obs, self.get_global_state(), -int(done), done, infos

        return obs, self.get_global_state(), 100, True, infos

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def get_env_info(self):
        output_dict = {}
        output_dict['n_actions'] = self.action_space[0].n
        output_dict['obs_shape'] = self.obs_dim
        output_dict['n_agents'] = self.nagents
        output_dict['state_shape'] = self.obs_dim
        output_dict['episode_limit'] = self.time_limit

        return output_dict


# def make_football_env(seed_dir, dump_freq=1000, representation='extracted', render=False):
def make_football_env_11vs4(dense_reward=False, dump_freq=1000):
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
    env = make_football_env_11vs4()
    obs, state = env.reset()
    print(obs.shape, state.shape)
