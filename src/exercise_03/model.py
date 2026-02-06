import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), # Capa de entrada con 1 neurona y capa oculta con 128 neuronas
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # 1 salida
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = MLP(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass
