import torch
import torch.nn as nn


class MLP(nn.Module):
    # Las imagenes del datset son de 3 canales (RGB) y 32x32 pixeles, y el output es de 10 clases
    def __init__(self, input_dim=3*32*32, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x): 
        x = x.view(x.size(0), -1)  # Flatten
        return self.net(x)


if __name__ == "__main__":
    model = MLP()
    print(model)
    x = torch.randn(1, 3, 32, 32) # Batch size 1, 3 canales, 32x32 píxeles
    print(model(x).shape) 
