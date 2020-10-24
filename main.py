# This the Python script to train the VAE model.

import torch
import numpy as np
from vae_net import VAE
import argparse
from tqdm import tqdm
import time
import scipy.io
import random
import math

from utils.test import test_function


def load_binary_mnist(d_dir):
    train_file = d_dir + '/BinaryMNIST/binarized_mnist_train.amat'
    valid_file = d_dir + '/BinaryMNIST/binarized_mnist_valid.amat'
    test_file = d_dir + '/BinaryMNIST/binarized_mnist_test.amat'
    mnist_train = np.concatenate([np.loadtxt(train_file), np.loadtxt(valid_file)])
    mnist_test = np.loadtxt(test_file)
    return mnist_train, mnist_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo for Training VAE")
    parser.add_argument("--dataset", default='mnist', dest='dataset',
                        choices=('mnist', 'ff'),
                        help="Dataset to train the VAE")
    parser.add_argument("--data_dir", dest='data_dir', default="../../dataset",
                        help="The directory of your dataset")
    parser.add_argument("--epochs", dest='num_epochs', default=102, type=int,
                        help="Total number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=100, type=int,
                        help="The batch size")
    parser.add_argument('--device', dest='device', default=0, type=int,
                        help='Index of device')
    parser.add_argument("--save_dir", dest='save_dir', default="./saved_models",
                        help="The directory to save your trained model")
    parser.add_argument('--decoder_type', dest='decoder_type', default='bernoulli', type=str,
                        help='Type of your decoder', choices=('bernoulli', 'gaussian'))
    parser.add_argument("--Nz", default=20, type=int,
                        help="Nz (dimension of the latent code)")
    args = parser.parse_args()

    # load data
    if args.dataset == 'mnist':
        dim = 28 * 28
        hid_num = 500
        train_num = 60000
        test_num = 10000
        if args.decoder_type == 'bernoulli':
            training_set, test_set = load_binary_mnist(args.data_dir)
        else:
            raise Exception("This implementation only provides Bernoulli decoder for MNIST")
    else:
        dim = 560
        hid_num = 200
        train_num = 1684
        test_num = 281
        ff = scipy.io.loadmat(args.data_dir + '/Frey_Face/frey_rawface.mat')['ff'].transpose() / 256
        test_index = random.sample([i for i in range(1965)], 281)
        train_index = list(set([i for i in range(1965)]) - set(test_index))
        training_set = ff[train_index, :]
        test_set = ff[test_index, :]
        if args.decoder_type == 'bernoulli':
            raise Exception("Can't use Bernoulli decoder on Frey Face")

    print(training_set.shape)
    print(test_set.shape)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    # define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = VAE(args, d=dim, h_num=hid_num)
    net.to(device)
    optimizer = torch.optim.Adagrad(net.parameters())

    # train the model
    start = time.time()
    for epoch in range(args.num_epochs):
        # test
        if epoch % 10 == 1:
            test_elbo = test_function(net, test_num, dataset=args.dataset, out_type=args.decoder_type,
                                      testset=test_set, device=device)
            print('test average ELBO=', test_elbo)

        # iterations
        running_loss = 0.0
        with tqdm(total=len(train_loader.dataset)) as progress_bar:
            for i, data in enumerate(train_loader, 0):
                train = data.to(device)
                optimizer.zero_grad()

                output = net(train.float())

                # the negative KL term
                negative_KL = (torch.ones_like(output[1]) + 2 * output[3] - output[1] * output[1] - torch.exp(
                    2 * output[3])).sum(1) / 2

                # the log conditional prob term
                if args.decoder_type == 'gaussian':
                    train_minus_mu = train - output[0]
                    log_p_x_given_z = -torch.ones_like(train).sum(1) * np.log(2 * math.pi) / 2 \
                                      - output[2].sum(1) / 2 - (
                                              train_minus_mu * train_minus_mu / (2 * torch.exp(output[2]))).sum(1)
                else:
                    log_p_x_given_z = torch.sum(output[0] * train - torch.log(1 + torch.exp(output[0])), 1)

                # update parameters
                loss = -negative_KL.mean() - log_p_x_given_z.mean()
                loss.backward()
                optimizer.step()
                running_loss -= negative_KL.sum().item()
                running_loss -= log_p_x_given_z.sum().item()

                # progress bar
                progress_bar.set_postfix(loss=loss.mean().item())
                progress_bar.update(data.size(0))

        print('[%d] loss: %.3f' % (epoch + 1, running_loss / train_num))

    print('Finished Training, time cost', time.time() - start)

    PATH = args.save_dir + '/vae_Nz_' + str(
        args.Nz) + '_dataset_' + args.dataset + '_decoder_' + args.decoder_type + '.pth'
    torch.save(net.state_dict(), PATH)
