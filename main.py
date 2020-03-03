import argparse
import torch
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from models import VariationalAutoencoder

parser = argparse.ArgumentParser(description='Variational Autoencoder implementation with AEVB and Wake-Sleep Training')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='neural network hidden layer dimension')
parser.add_argument('--z_dim', type=int, default=32,
                    help='latent space dimensionality')
parser.add_argument('--algo', type=str, default="vae", choices=["vae", "wake-sleep"],
                    help='training algorithm (either vae or wake-sleep)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = MNIST("", train=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_data = MNIST("", train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

input_shape = train_data.__getitem__(0)[0].shape
input_dim = input_shape[0] * input_shape[1] * input_shape[2]

model = VariationalAutoencoder(input_dim, args.hidden_dim, args.z_dim, args.algo, device)


for e in range(args.epochs):
    print("Training Epoch:", e)
    #Train model
    for images, _ in train_loader:
        images = images.to(device)
        model.train(images)
    #Test model
    x, _ = test_data.__getitem__(0)
    x = x.to(device)
    x_hat, _, _ = model(x)
    comparison = torch.cat([x, x_hat.view(1, 28, 28)], axis=2)
    vutils.save_image(comparison.cpu(), 'reconstruction_' + str(e) + '.png')
    #    images = images.to(device)