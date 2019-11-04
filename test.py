import random
import torch
from run import PPO
from torch.distributions import Categorical
from simple_env import SimpleEnv

PATH = 'PPO.pth'
model = PPO()
model.load_state_dict(torch.load(PATH))

env_test = SimpleEnv(testing=True)
ob = env_test.reset(1, 3)


for _ in range(100):
    score = 0
    while(True):
        # action = policy(ob)
        prob_1 = model.pi_1(ob[0].float())
        a_1 = prob_1.argmax().item()
        print(a_1)
        
        prob_2 = model.pi_2(ob[0].float())
        a_2 = prob_2.argmax().item()
        action = [a_1, a_2]
        ob,r, episode_over, _ = env_test.step(action)
        print(r)
        score += r
        if episode_over:
            break

    print(r)
    env_test.render()
    env_test.reset()
