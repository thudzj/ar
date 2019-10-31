import gym
import gym_auto
import random

env = gym.make('gym_auto:auto-v0')
env.reset()

print(env.action_space)
print(env.observation_space)

c = 3
d = 1

for _ in range(500):
    #a = random.randint(1,7) - 1
    #b = random.randint(1,4) - 1
    env.step([c,d])
    #env.step([a,b])
env.render()


