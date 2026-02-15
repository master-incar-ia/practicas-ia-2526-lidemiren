import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Primera capa convolucional: 3 canales de entrada (RGB), 64 filtros, tamaño de kernel 3x3, padding de 1 para mantener el tamaño de la imagen.
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            # Capa de normalización por lotes para estabilizar y acelerar el entrenamiento.
            nn.BatchNorm2d(64),
            # Función de activación ReLU para introducir no linealidad.
            nn.ReLU(inplace=True),
            # Segunda capa convolucional: 64 canales de entrada, 64 filtros, tamaño de kernel 3x3, padding de 1.
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Capa de maxpooling para reducir la dimensionalidad espacial de las características extraídas por las capas convolucionales. 
            # Para reducir el número de parámetros y a controlar el sobreajuste.
            nn.MaxPool2d(kernel_size=2, stride=2), 

            # Tercera capa convolucional: 64 canales de entrada, 128 filtros, tamaño de kernel 3x3, padding de 1.
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Cuarta capa convolucional: 128 canales de entrada, 128 filtros, tamaño de kernel 3x3, padding de 1.
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Segunda capa de maxpooling para reducir aún más la dimensionalidad espacial y controlar el sobreajuste.
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Quinta capa convolucional: 128 canales de entrada, 256 filtros, tamaño de kernel 3x3, padding de 1.
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Tercera capa de maxpooling para reducir la dimensionalidad
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Dropout(0.5), 
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x  
    

if __name__ == "__main__":
    model = CNN(1, 1)
    print(model)
    x = torch.tensor([1.0])
    print(model(x))
    pass