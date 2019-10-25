import numpy as np
import torch
from tasks import Task, random_task
from optimizers import OPTS, LRS, GradStats

if __name__ == "__main__":
    # initialize a task
    task = Task()
    T = 5 # max steps of the inner unrollment

    # define variables
    w = torch.tensor([-2.], requires_grad=True)
    alpha = torch.tensor([2.], requires_grad=True)

    # define optimizer
    optimizer_alpha = torch.optim.SGD([alpha], lr=.3)

    # initialize the statistics of gradients
    grad_stats = GradStats()

    # initialize the policy network and the corresponding optimizer
    # policy_net = LSTMcell(20)
    # policy_h = torch.tensor(20)
    # optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=.01)

    o_loss = torch.tensor([0.])
    o_grad = torch.tensor([0.])

    for outer_ite in range(1000):

        ws = [torch.tensor(w.data)]
        for inner_ite in range(T):
            i_loss = task.inner_loss(ws[-1], alpha)
            i_grad = task.inner_grad(ws[-1], alpha)
            grad_stats.update(i_grad)

            # choose optimizer and lr with a policy net
            # policy_o, policy_h = policy_net(torch.concat([o_loss, o_grad, i_loss, i_grad], 0), policy_h)
            # opt, lr = policy_o.sample()
            # logp = logp(opt | policy_o) + logp(lr | policy_o)
            opt, lr = 0, 0
            ws.append(OPTS[opt](LRS[lr], ws[-1], i_grad, grad_stats))

        w = ws[-1]

        o_loss = task.outer_loss(w, alpha)
        optimizer_alpha.zero_grad()
        o_loss.backward()
        o_grad = alpha.grad.clone().detach()
        optimizer_alpha.step()

        # get reward and update the policy
        # reward = task.get_reward(alpha)
        # optimizer_policy.zero_grad()
        # policy_loss = -logp * reward
        # policy_loss.backward()
        # optimizer_policy.step()

        print(alpha, w)
