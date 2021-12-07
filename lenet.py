import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.process = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=6,
                      kernel_size=5),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Sigmoid(),
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

        pass

    def forward(self, x):
        return self.process(x)


def main():
    model = LeNet(num_classes=10)
    sample = torch.randn((5, 1, 32, 32))
    output = model(sample)
    print(output.shape)


if __name__ == "__main__":
    main()
