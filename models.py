import torch
import torch.optim as optim
import torch.nn as nn

class VariationalAutoencoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, z_dim, algo, device):
        super(VariationalAutoencoder, self).__init__()
        self.algo = algo
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2*self.z_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim)
        )
        
        self.vae_optimizer = optim.Adam(self.parameters())
        self.wake_optimizer = optim.Adam(self.decoder.parameters(), lr=1e-3)
        self.sleep_optimizer = optim.Adam(self.encoder.parameters(), lr=1e-5)
        
    def encode(self, x):
        z_mu, z_logvar = self.encoder(x).split(self.z_dim, 1)
        return z_mu, z_logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma
        
    def forward(self, x):
        z_mu, z_logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(z_mu, z_logvar)
        x_hat = self.decode(z).view(-1, 1, 28, 28)
        return x_hat, z_mu, z_logvar
    
    def train(self, x):
        if self.algo == "vae":
            self.vae_optimizer.zero_grad()
            x_hat, z_mu, z_logvar = self.forward(x)
            recon_loss = torch.sum((x-x_hat)**2)
            kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu**2 - torch.exp(z_logvar))
            vae_loss = kl_loss + recon_loss
            vae_loss.backward()
            self.vae_optimizer.step()
        else:
            #Optimize Wake objective
            self.wake_optimizer.zero_grad()
            wake_loss = 0.0
            for l in range(10):
                x_hat, _, _ = self.forward(x)
                wake_loss += 1/10 * torch.sum((x-x_hat)**2)
            wake_loss.backward()
            self.wake_optimizer.step()
            
            #Optimize Sleep objective
            self.sleep_optimizer.zero_grad()
            sleep_loss = 0.0
            for l in range(10):
                z = torch.randn(self.z_dim).to(self.device)
                x_decoded = self.decode(z)
                x_hat, z_mu, z_logvar = self.forward(x_decoded)
                dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=z_mu, covariance_matrix=torch.diag(torch.exp(z_logvar.squeeze(0))))
                logprob = dist.log_prob(z)
                sleep_loss -= 1/10 * logprob
            sleep_loss.backward()
            self.sleep_optimizer.step()