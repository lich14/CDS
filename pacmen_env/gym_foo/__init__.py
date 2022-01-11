from gym.envs.registration import register

register(
    id='pacmen-tini-4ag-v0',
    entry_point='gym_foo.envs:CustomEnv',
    kwargs={
        "n_agents": 4,
        "mode": 'tini',
    },
)

register(
    id='pacmen-small-4ag-v0',
    entry_point='gym_foo.envs:CustomEnv',
    kwargs={
        "n_agents": 4,
        "mode": 'small',
    },
)

register(
    id='pacmen-large-4ag-v0',
    entry_point='gym_foo.envs:CustomEnv',
    kwargs={
        "n_agents": 4,
        "mode": 'large',
    },
)