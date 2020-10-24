import torch
import numpy as np
import math


def test_function(net, test_n, dataset, out_type, testset, device='cpu'):
    if dataset == 'ff':
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    else:
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)
    elbo_total = 0.0
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            test = data.to(device)
            output = net(test.float())

            # the negative KL term
            negative_kl = (torch.ones_like(output[3]) + 2 * output[3] - output[1] * output[1] - torch.exp(
                2 * output[3])).sum(1) / 2

            # the log conditional prob term
            if out_type == 'gaussian':
                test_minus_mu = test - output[0]
                log_p_x_given_z = -torch.ones_like(test).sum(1) * np.log(2 * math.pi) / 2 - output[2].sum(1) / 2 - (
                        test_minus_mu * test_minus_mu / (2 * torch.exp(output[2]))).sum(1)
            else:
                log_p_x_given_z = torch.sum(output[0] * test - torch.log(1 + torch.exp(output[0])), 1)

            # sum of the variational lower bounds
            elbo = negative_kl.sum() + log_p_x_given_z.sum()
            elbo_total += elbo.item()
    return elbo_total / test_n
