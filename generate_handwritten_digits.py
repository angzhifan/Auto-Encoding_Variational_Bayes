import torch
from vae_net import VAE
import argparse
import matplotlib.pyplot as plt


def generate(net, z, out_type):
    x = torch.tanh(net.fc3(z))
    if out_type == 'gaussian':
        out = torch.sigmoid(net.fc4_mu(x))
    else:
        out = torch.sigmoid(net.fc4(x))
    return out.detach().view(28, 28)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Handwritten Digits")
    parser.add_argument("--model_dir", dest='model_dir',
                        default="./saved_models",
                        help="The directory and name of your model")
    parser.add_argument("--Nz", default=20, type=int,
                        help="Nz (dimension of the latent code)")
    parser.add_argument('--decoder_type', dest='decoder_type', default='bernoulli', type=str,
                        help='Type of your decoder', choices=('bernoulli', 'gaussian'))
    parser.add_argument("--num_row", dest='n_row',
                        default=10, help="The number of rows you want")
    parser.add_argument("--num_col", dest='n_col',
                        default=10, help="The number of columns you want")
    args = parser.parse_args()

    dim = 28 * 28
    hid_num = 500

    filename = args.model_dir + "/vae_Nz_" + str(args.Nz) + "_dataset_mnist_decoder_" + args.decoder_type + ".pth"
    if args.decoder_type == 'gaussian':
        net = VAE(args, d=dim, h_num=hid_num)
        net.load_state_dict(
            torch.load(filename, map_location=torch.device('cpu')))
    elif args.decoder_type == 'bernoulli':
        net = VAE(args, d=dim, h_num=hid_num)
        net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    n_row, n_col = args.n_row, args.n_col
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_row, n_col))
    for i in range(args.n_row):
        for j in range(n_col):
            z1 = torch.randn(args.Nz)
            img = generate(net, z1, args.decoder_type)
            np_img = img.numpy()
            axes[i, j].imshow(np_img)

    plt.savefig("vae_Nz_" + str(args.Nz) + "_decoder_" + args.decoder_type)
