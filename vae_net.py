"""
Auto-Encoding Variational Bayes Network Architecture
"""

import torch
import torch.nn as nn


# VAE with one stochastic layer z
class VAE(nn.Module):

    def __init__(self, args, d, h_num, scaled=True):
        super(VAE, self).__init__()
        self.dim = d
        self.Nz = args.Nz
        self.hid_num = h_num
        self.output_type = args.decoder_type
        self.scaled_mean = scaled
        self.fc1 = nn.Linear(d, h_num)
        self.fc2_mu = nn.Linear(h_num, args.Nz)
        self.fc2_sigma = nn.Linear(h_num, args.Nz)
        self.fc3 = nn.Linear(args.Nz, h_num)
        if args.decoder_type == 'gaussian':
            self.fc4_mu = nn.Linear(h_num, d)
            self.fc4_sigma = nn.Linear(h_num, d)
        else:
            self.fc4 = nn.Linear(h_num, d)

    def forward(self, x):
        x = x.view(-1, self.dim)
        x = torch.tanh(self.fc1(x))
        mu_z = self.fc2_mu(x)
        log_sigma_z = self.fc2_sigma(x)
        eps = torch.randn_like(mu_z)
        x = mu_z + torch.exp(log_sigma_z) * eps
        x = torch.tanh(self.fc3(x))
        if self.output_type == 'gaussian':
            if self.scaled_mean:
                mu = torch.sigmoid(self.fc4_mu(x))
            else:
                mu = self.fc4_mu(x)
            log_sigma = self.fc4_sigma(x)
            return mu, mu_z, log_sigma, log_sigma_z
        else:
            x = self.fc4(x)
            return x, mu_z, '_', log_sigma_z
