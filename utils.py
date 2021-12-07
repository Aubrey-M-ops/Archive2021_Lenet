import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

data_location = "./data"


def get_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size, shuffle)


def get_dataset(name="mnist"):
    assert name in ['mnist']
    if name == 'mnist':
        train = datasets.MNIST(root=data_location,
                               download=True,
                               transform=T.Compose([
                                   T.Resize((32, 32)),
                                   T.ToTensor(),
                                   T.Normalize(0.1370, 0.3081)
                               ]),
                               train=True)
        test = datasets.MNIST(root=data_location,
                              download=True,
                              transform=T.Compose([
                                  T.Resize((32, 32)),
                                  T.ToTensor(),
                                  T.Normalize(0.1370, 0.3081)
                              ]),
                              train=False)
    return train, test


class AverageMeter():
    def __init__(self):
        self.avg = 0
        self.sumv = 0
        self.cnt = 0
        self.reset()
      
    def reset(self):
        self.avg = 0
        self.sumv = 0
        self.cnt = 0
      
    def update(self, val, n=1):
        self.sumv += val * n
        self.cnt += n
        self.avg = self.sumv / self.cnt