# Simple generator and Discriminator models with just 1 hidden layer. Add more layers for better performance.

# Generator Network
class Generator(nn.Module):
    # declare layers
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(5, 100)
        self.output = nn.Linear(100, 1)

    # Forward: one ReLU hidden layer of 400 nodes, one Sigmoid output layer of 1 nodes
    def forward(self, z):
        h = nn.functional.softplus(self.fc1(z), beta=10)
        out = nn.functional.softplus(self.output(h), beta=10)
        return out

# The discriminator will take all option related values (model params and volatility)
# and decide if it is generated

# Discriminator Network
class Discriminator(nn.Module):

    # declare layers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(6, 100)
        self.output = nn.Linear(100, 1)

    # Forward: one ReLU hidden layer of 400 nodes, one Sigmoid output layer of 1 node
    def forward(self, z):
        h = nn.functional.softplus(self.fc1(z), beta=20)
        out = self.output(h)
        return out