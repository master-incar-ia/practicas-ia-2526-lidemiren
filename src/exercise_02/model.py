import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.net(x)
        return x


if __name__ == "__main__":
    model = MLP(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass