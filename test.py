import random
import torch
from run import PPO
from torch.distributions import Categorical
from simple_env import SimpleEnv

PATH = 'PPO.pth'
model = PPO()
model.load_state_dict(torch.load(PATH))

env_test = SimpleEnv(1, 3, testing=True)
ob = env_test.reset(1, 3)


score = 0
while(True):
    # action = policy(ob)
    prob_1 = model.pi_1(ob[0].float())
    a_1 = prob_1.argmax().item()
    action = [a_1, 1]
    ob,r, episode_over, _ = env_test.step(action)
    score += r
    if episode_over:
        break

print(r)
env_test.render()
