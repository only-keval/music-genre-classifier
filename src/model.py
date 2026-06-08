import torch.nn as nn


class GenreCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)