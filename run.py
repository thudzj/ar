import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from simple_env import SimpleEnv

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 500
state_space   = 7
action_space  = 8
para_space    = 5

PATH = 'PPO.pth'

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(state_space,256)
        self.fc_pi_1 = nn.Linear(256,action_space)
        self.fc_pi_2 = nn.Linear(256,para_space)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi_1(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi_1(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def pi_2(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi_2(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_lst, done = transition

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_lst)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float),\
            torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi_1 = self.pi_1(s, softmax_dim=1)
            #print(pi_1)
            pi_a = pi_1.gather(1,a[:,0].reshape(a[:,0].size()[0],1))
            #print(pi_a)
            
            pi_2 = self.pi_2(s, softmax_dim=1)
            pi_b = pi_2.gather(1,a[:,1].reshape(a[:,1].size()[0],1))
            #pi_1 = self.pi_1(s, softmax_dim=1)
            #print(pi_1)
            #spi_a = pi_1.gather(1,a[:,1].reshape(a[:,1].size()[0],1))
            #print(pi_a)

            prob_n_1 = prob_a[:,0].reshape(prob_a[:,0].size()[0],1)
            prob_n_2 = prob_a[:,1].reshape(prob_a[:,1].size()[0],1)
            #print(prob_n)
            ratio = torch.exp(torch.log(pi_a) + torch.log(pi_b) - torch.log(prob_n_1) - torch.log(prob_n_2))  # a/b == exp(log(a)-log(b))

            #ratio = torch.exp(torch.log(pi_a) - torch.log(prob_n_1) )  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            #print(self.v(s))
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            #print(loss)

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def save_net(self):
        print("save_net")
        torch.save(self.state_dict(),PATH)


def main():
    env = SimpleEnv(seed=12345)
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
            #print(s[0])
                prob_1 = model.pi_1(s[0].float())
                # print(prob_1)
                m_1 = Categorical(prob_1)
                #print(m_1)

                a_1 = m_1.sample().item()

                prob_2 = model.pi_2(s[0].float())
                m_2 = Categorical(prob_2)

                a_2 = m_2.sample().item()
                #a_2 = 0
                a = [a_1, a_2]
                s_prime, r, done, info = env.step(a)
                

                prob_lst = [prob_1[a_1].item(), prob_2[a_2].item()]
                model.put_data((s[0].numpy(), a, r, s_prime[0].numpy(), prob_lst, done))
                s = s_prime

                score += r

                if done:
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            model.save_net()
            score = 0.0

    env.close()
'''
    env_test = SimpleEnv(1, 3, testing=True)
    ob = env.init_ob
    while(True):
        # action = policy(ob)
        action = (6, 1)
        ob, reward, episode_over, tmp = env_test.step(action)
        if episode_over:
            break
    env_test.render()
'''

if __name__ == '__main__':
    main()
