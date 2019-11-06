import random
import torch
from run import PPO
from torch.distributions import Categorical
from simple_env import SimpleEnv

PATH = 'PPO_1.0.pth'
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
        
        prob_2 = model.pi_2(ob[0].float())
        a_2 = prob_2.argmax().item()
        action = [a_1, a_2] #[2, 0] if env_test.inner_count < env_test.inner_T - 1 else [7, 0]
        print(action)
        ob,r, episode_over, _ = env_test.step(action)
        print(r)
        score += r
        if episode_over:
            break

    print(r)
    env_test.render()
    env_test.reset()
