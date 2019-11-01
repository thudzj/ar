import random
from simple_env import SimpleEnv

env = SimpleEnv()

for _ in range(1):
    while(True):
        ob, reward, episode_over, tmp = env.step([0,1])
        if episode_over:
            break
    env.render()
    env.reset()

env_test = SimpleEnv(1, 3, testing=True)
while(True):
    ob, reward, episode_over, tmp = env_test.step([6,1])
    if episode_over:
        break
env_test.render()
