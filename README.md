# Auto-Encoding_Variational_Bayes

A Pytorch Implementation of the paper *Auto-Encoding Variational Bayes* by Diederik P. Kingma and Max Welling.
https://arxiv.org/abs/1312.6114

## Usage

```
usage: main.py [-h] [--dataset {mnist,ff}] [--data_dir DATA_DIR]
               [--epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
               [--device DEVICE] [--save_dir SAVE_DIR]
               [--decoder_type {bernoulli,gaussian}] [--Nz NZ]

Demo for Training VAE

optional arguments:
  -h, --help            show this help message and exit
  --dataset {mnist,ff}  Dataset to train the VAE
  --data_dir DATA_DIR   The directory of your dataset
  --epochs NUM_EPOCHS   Total number of epochs
  --batch_size BATCH_SIZE
                        The batch size
  --device DEVICE       Index of device
  --save_dir SAVE_DIR   The directory to save your trained model
  --decoder_type {bernoulli,gaussian}
                        Type of your decoder
  --Nz NZ               Nz (dimension of the latent code)
```

It can be seen by running
```
python main.py --help
```

## Datasets 

Binarized MNIST and Frey Face

The binarized MNIST dataset can be downloaded from 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat 
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat

The Frey Face dataset can be downloaded from
https://cs.nyu.edu/~roweis/data.html
