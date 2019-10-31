from gym.envs.registration import register

register(id='auto-v0', entry_point='gym_auto.envs:AutoEnv',)
register(id='auto-complex-v0', entry_point='gym_auto.envs:AutoComplexEnv',)
