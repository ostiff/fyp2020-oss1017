# TODO: Document code

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
import random

class Autoencoder(nn.Module):
    def __init__(self, input_size, layers, latent_dim=2, device="cpu"):
        super(Autoencoder, self).__init__()

        self.device=device
        self.input_size = input_size
        self.latent_dim = latent_dim

        enc = []
        prev = input_size
        for l in layers:
            enc.append(nn.Linear(prev, prev:=l))
            enc.append(nn.Sigmoid())
        enc.append(nn.Linear(prev, latent_dim))
        enc.append(nn.Sigmoid())

        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for l in reversed(layers):
            dec.append(nn.Linear(prev, prev:=l))
            dec.append(nn.Sigmoid())

        dec.append(nn.Linear(prev, input_size))
        dec.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*dec)


    def forward(self, x):
        return self.decoder(self.encoder(x.view(-1, self.input_size)))

    @staticmethod
    def loss_fn(recon_x, x):
        return F.mse_loss(recon_x, x, reduction='sum')

    @torch.no_grad()
    def encode_inputs(self, loader):
        z = []

        for i, data in enumerate(loader):
            x = data.to(self.device).float().view(-1, self.input_size)
            x = self.encoder(x)
            z.append(x)

        return torch.cat(z, dim=0).to("cpu").numpy()


def train_autoencoder(model, optimizer, loader_train, loader_test, num_epochs, **anim):
    state_np = np.random.get_state()
    state_py = random.getstate()
    state_torch = torch.random.get_rng_state()
    if 'animation_data' in anim:
        animation_enable = True
        animation_data = anim['animation_data']

        if 'animation_colour' in anim:
            animation_colour = anim['animation_colour']
        else:
            animation_colour = [['#539ecd'] * len(animation_data.dataset)]

        n_features = len(animation_colour)
        cols = math.ceil(math.sqrt(n_features))
        rows = math.ceil(n_features / cols)

        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(2*cols,2*rows))
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        axes = axes.flatten()

        if 'animation_labels' in anim:
            animation_labels = anim['animation_labels']
        else:
            animation_labels = [f'Feature {i}' for i in range(n_features)]

        for i, ax in enumerate(axes):
            if i < n_features:
                ax.axis('equal')
                ax.axis([-3, 1, -3, 1])
                ax.title.set_text(animation_labels[i])
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.axis('off')

        frames = []

    else:
        animation_enable = False
    random.setstate(state_py)
    np.random.set_state(state_np)
    torch.random.set_rng_state(state_torch)


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


            if animation_enable:
                state_np = np.random.get_state()
                state_py = random.getstate()
                state_torch = torch.random.get_rng_state()

                encoded = model.encode_inputs(animation_data)
                #encoded = np.random.rand(len(animation_colour[0]),2)
                frame = []

                title = plt.figtext(0.5,0.97, "Epoch: {0:4}".format(epoch),
                                size=plt.rcParams["axes.titlesize"],
                                ha="center")
                frame.append(title)

                for i, ax in enumerate(axes):
                    if i < n_features:
                        ax.axis('equal')
                        ax.axis([-3, 1, -3, 1])
                        scat = ax.scatter(encoded[:, 0], encoded[:, 1], c=animation_colour[i], s=2)
                        frame.append(scat)

                frames.append(frame)

                random.setstate(state_py)
                np.random.set_state(state_np)
                torch.random.set_rng_state(state_torch)

    ret = {'train_loss': train_losses,
            'test_loss': test_losses
           }

    if animation_enable:
        anim_obj = animation.ArtistAnimation(fig, frames, interval=50, blit=False)

        if 'animation_path' in anim:
            anim_obj.save(anim['animation_path'], fps=15)

        # ret['gif'] = anim_obj.to_jshtml()
        plt.clf()


    return ret


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
