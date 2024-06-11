from torch import nn

class autoencoder(nn.Module):
    def __init__(self ,input_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 20),
            nn.ReLU(True),
            nn.Linear(20, 10),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.ReLU(True), 
            nn.Linear(5, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(True),
            nn.Linear(5, 10),
            nn.ReLU(True),
            nn.Linear(10, 20),
            nn.ReLU(True),
            nn.Linear(20, input_size), 
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x