from gym.envs.registration import register

register(
    id='quad-v0',
    entry_point='gym_quadrotor.envs:QuadrotorEnv',
)
