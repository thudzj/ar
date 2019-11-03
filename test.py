import random
import torch
from run import PPO
from torch.distributions import Categorical
from simple_env import SimpleEnv

PATH = 'PPO.pth'
model = PPO()
model.load_state_dict(torch.load(PATH))

env_test = SimpleEnv(1, 3, testing=True)
ob = env_test.reset()


while(True):
    # action = policy(ob)
    prob_1 = model.pi_1(ob[0].float())
    m_1 = Categorical(prob_1)
    print(m_1)
    a_1 = m_1.sample().item()
    action = [a_1, 1]
    ob,episode_over, _, _ = env_test.step(action)
    if episode_over:
        break

env_test.render()
