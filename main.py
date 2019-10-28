import numpy as np
import torch
from tasks import Task, random_task
from optimizers import OPTS, LRS, GradStats

if __name__ == "__main__":
    # hyper-parameters
    weight_decay_w = 0
    beta2_rmsprop = 0.9
    outer_T = 100
    inner_T = 5

    # initialize a task
    task = Task()

    # define variables
    w = torch.tensor([-2.], requires_grad=True)
    alpha = torch.tensor([2.], requires_grad=True)

    # define optimizer
    optimizer_alpha = torch.optim.SGD([alpha], lr=.3)

    # initialize the statistics of gradients
    grad_stats = GradStats(w, beta2_rmsprop=beta2_rmsprop)

    # initialize the policy network and the corresponding optimizer
    # policy_net = MLP(20)
    # optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=.01)

    o_loss = torch.tensor([0.])
    o_grad = torch.tensor([0.])

    for outer_ite in range(outer_T):

        grad_stats.detach()
        ws = [w.detach_().requires_grad_()]
        for inner_ite in range(inner_T):
            i_loss = task.inner_loss(ws[-1], alpha) + weight_decay_w/2. * (ws[-1])**2
            i_grad = torch.autograd.grad(outputs=i_loss, inputs=ws[-1], grad_outputs=torch.ones_like(i_loss),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_stats.update(i_grad)

            # choose optimizer and lr with a policy net
            # policy_o = policy_net(torch.concat([o_loss, o_grad, i_loss, i_grad], 0))
            # opt, lr = policy_o.sample()
            # logp = logp(opt | policy_o) + logp(lr | policy_o)
            opt, lr = 6, 1
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

        print("Training ite", outer_ite, alpha, w)

    ## test phase: test the trained policy network
    print("Testing")
    task_new = random_task()
    for outer_ite in range(outer_T):

        grad_stats.detach()
        ws = [w.detach_().requires_grad_()]
        for inner_ite in range(inner_T):
            i_loss = task_new.inner_loss(ws[-1], alpha) + weight_decay_w/2. * (ws[-1])**2
            i_grad = torch.autograd.grad(outputs=i_loss, inputs=ws[-1], grad_outputs=torch.ones_like(i_loss),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_stats.update(i_grad)

            # choose optimizer and lr with a policy net
            # policy_o = policy_net(torch.concat([o_loss, o_grad, i_loss, i_grad], 0))
            # opt, lr = policy_o.sample()
            # logp = logp(opt | policy_o) + logp(lr | policy_o)
            opt, lr = 6, 1
            ws.append(OPTS[opt](LRS[lr], ws[-1], i_grad, grad_stats))

        w = ws[-1]

        o_loss = task_new.outer_loss(w, alpha)
        optimizer_alpha.zero_grad()
        o_loss.backward()
        o_grad = alpha.grad.clone().detach()
        optimizer_alpha.step()
    print("Optima: ", task_new.optimal_alpha, task_new.optimal_w)
    print("Found solution: ", alpha.item(), w.item())
