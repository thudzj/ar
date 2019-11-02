import random
from simple_env import SimpleEnv

env = SimpleEnv(seed=12345)
ob = env.init_ob

for _ in range(2):
    while(True):
        # action = policy(ob)
        action = (0, 1)
        ob, reward, episode_over, tmp = env.step(action)
        # update the policy with reward
        if episode_over:
            break
    env.render()

    env.reset()
    ob = env.init_ob

env_test = SimpleEnv(1, 3, testing=True)
ob = env.init_ob
while(True):
    # action = policy(ob)
    action = (6, 1)
    ob, reward, episode_over, tmp = env_test.step(action)
    if episode_over:
        break
env_test.render()
