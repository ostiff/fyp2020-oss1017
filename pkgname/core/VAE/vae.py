# TODO: Document code

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class VAE(nn.Module):
    def __init__(self, device="cpu", latent_dim=2, input_size=10, h_dim=5):
        super(VAE, self).__init__()

        self.device=device

        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_size, h_dim),
            #nn.ReLU(),

            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, 2 * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_size),
            nn.Sigmoid()
        )


    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        x = x.view(-1, self.input_size)
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


    def loss_fn(self, recon_x, x, mu, logvar, beta):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean') * self.input_size
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

        return recon_loss + beta * KLD, recon_loss, KLD


    def encode_inputs(self, loader):
        self.eval()
        z = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                mu, logvar = self.encode(data.to(self.device).float())
                z.append(self.reparametrize(mu, logvar))

        self.train()

        return torch.cat(z, dim=0).to("cpu").numpy()


def train_vae(model, optimizer, loader_train, loader_test, num_epochs, beta):
    model.train()
    train_losses, train_recon, train_KLD = [], [], []
    test_losses, test_recon, test_KLD = [], [], []

    for epoch in range(num_epochs):

        train_loss, recon_loss, KL_loss = 0, 0, 0

        # TRAIN
        for i, data in enumerate(loader_train):
            model.train()
            data = data.to(model.device).float()

            out, mu, logvar = model(data)
            loss, recon, KLD = model.loss_fn(out, data, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            recon_loss += recon.item()
            KL_loss += KLD.item()

        # DISPLAY/SAVE TRAINING LOSSES
        print("TRAIN: \tEpoch {0:2}, Total loss:{1:8.4f}, Recon loss:{2:8.4f}, KL loss:{3:8.4f}".format(
            epoch,
            train_loss / len(loader_train),
            recon_loss / len(loader_train),
            KL_loss / len(loader_train))
        )

        train_losses.append(train_loss / len(loader_train))
        train_KLD.append(KL_loss / len(loader_train))
        train_recon.append(recon_loss / len(loader_train))

        # DISPLAY/SAVE TESTING LOSSES
        model.eval()
        with torch.no_grad():
            test_loss, test_recon_loss, test_KL_loss = 0, 0, 0
            for i, data in enumerate(loader_test):
                data = data.to(model.device).float()

                out, mu, logvar = model(data)
                total, recon, KLD = model.loss_fn(out, data, mu, logvar, beta)

                test_loss += total.item()
                test_KL_loss += KLD.item()
                test_recon_loss += recon.item()

            print("TEST: \tEpoch {0:2}, Total loss:{1:8.4f}, Recon loss:{2:8.4f}, KL loss:{3:8.4f}".format(
                epoch,
                test_loss / len(loader_test),
                test_recon_loss / len(loader_test),
                test_KL_loss / len(loader_test))
            )

            test_losses.append(test_loss / len(loader_test))
            test_KLD.append(test_KL_loss / len(loader_test))
            test_recon.append(test_recon_loss / len(loader_test))

    return {'train_loss': train_losses,
            'train_kld': train_KLD,
            'train_recon': train_recon,
            'test_loss': test_losses,
            'test_kld': test_KLD,
            'test_recon': test_recon
           }


def plot_vae_loss(losses, show=False, printout=True, dpi=300, fname='./vae_losses.jpg'):
    fig1 = plt.figure(constrained_layout=True, figsize=(12, 8))
    spec1 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig1)

    num_epochs = range(len(losses['train_loss']))

    f1_ax11 = fig1.add_subplot(spec1[0, 0])
    f1_ax11.set_title('Train Total Loss')
    f1_ax11.plot(num_epochs, losses['train_loss'])

    f1_ax12 = fig1.add_subplot(spec1[0, 1])
    f1_ax12.set_title('Train Recon Loss')
    f1_ax12.plot(num_epochs, losses['train_recon'])

    f1_ax13 = fig1.add_subplot(spec1[0, 2])
    f1_ax13.set_title('Train KL Loss')
    f1_ax13.plot(num_epochs, losses['train_kld'])

    f1_ax21 = fig1.add_subplot(spec1[1, 0])
    f1_ax21.set_title('Test Total Loss')
    f1_ax21.plot(num_epochs, losses['test_loss'])

    f1_ax22 = fig1.add_subplot(spec1[1, 1])
    f1_ax22.set_title('Test Recon Loss')
    f1_ax22.plot(num_epochs, losses['test_recon'])

    f1_ax23 = fig1.add_subplot(spec1[1, 2])
    f1_ax23.set_title('Test KL Loss')
    f1_ax23.plot(num_epochs, losses['test_kld'])

    if printout:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)

    if show:
        plt.show()


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
