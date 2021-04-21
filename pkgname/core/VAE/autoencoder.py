# TODO: Document code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Autoencoder(nn.Module):
    def __init__(self, input_size, layers, latent_dim=2, device="cpu"):
        super(Autoencoder, self).__init__()

        self.device=device
        self.input_size = input_size
        self.latent_dim = latent_dim

        enc = []
        prev = input_size
        for i in range(len(layers)):
            enc.append(nn.Linear(prev, prev:=layers[i]))
            enc.append(nn.ReLU())
        enc.append(nn.Linear(prev, latent_dim))

        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for i in range(len(layers)-1, -1, -1):
            dec.append(nn.Linear(prev, prev:=layers[i]))
            dec.append(nn.ReLU())

        dec.append(nn.Linear(prev, input_size))
        dec.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec)


    def forward(self, x):
        return self.decoder(self.encoder(x.view(-1, self.input_size)))

    @staticmethod
    def loss_fn(recon_x, x):
        return F.mse_loss(recon_x, x, reduction='sum')

    def encode_inputs(self, loader):
        self.eval()
        z = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                x = self.encoder(data.to(self.device).float())
                z.append(x)

        self.train()

        return torch.cat(z, dim=0).to("cpu").numpy()


def train_autoencoder(model, optimizer, loader_train, loader_test, num_epochs):
    model.train()
    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        train_loss = 0

        # TRAIN
        for i, data in enumerate(loader_train):
            model.train()
            data = data.to(model.device).float()

            out = model(data)
            loss = model.loss_fn(out, data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # DISPLAY/SAVE TRAINING LOSSES
        print("TRAIN: \tEpoch {0:2}, Loss:{1:8.4f}".format(
            epoch,
            train_loss / len(loader_train))
        )

        train_losses.append(train_loss / len(loader_train))

        # DISPLAY/SAVE TESTING LOSSES
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for i, data in enumerate(loader_test):
                data = data.to(model.device).float()

                out = model(data)
                total = model.loss_fn(out, data)

                test_loss += total.item()

            print("TEST: \tEpoch {0:2}, Loss:{1:8.4f}".format(
                epoch,
                test_loss / len(loader_test))
            )

            test_losses.append(test_loss / len(loader_test))

    return {'train_loss': train_losses,
            'test_loss': test_losses
           }


def plot_autoencoder_loss(losses, show=False, printout=True, dpi=300, fname='./autoencoder_losses.jpg'):
    fig1 = plt.figure(constrained_layout=True, figsize=(12, 8))
    spec1 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig1)

    num_epochs = range(len(losses['train_loss']))

    f1_ax11 = fig1.add_subplot(spec1[0, 0])
    f1_ax11.set_title('Train Loss')
    f1_ax11.plot(num_epochs, losses['train_loss'])

    f1_ax12 = fig1.add_subplot(spec1[0, 1])
    f1_ax12.set_title('Test Loss')
    f1_ax12.plot(num_epochs, losses['test_loss'])

    plot = plt.gcf()

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()

    return plot

def get_device(usegpu=True):
    if usegpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print(f'Using {device}')

    return device


def set_seed(seed=0):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
