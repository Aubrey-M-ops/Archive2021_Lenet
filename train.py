import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from lenet import LeNet
from utils import AverageMeter, get_dataloader, get_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model_epoch(model,
                      dataloader,
                      loss_func,
                      optimizer,
                      current_epoch,
                      total_epoches,
                      log_freq=10):
    model.train()
    accu_meter = AverageMeter()
    loss_meter = AverageMeter()
    for current_batch_count, data in enumerate(dataloader):
        images, labels = data[0].to(device), data[1].to(device)
        output = model(images)
        loss_value = loss_func(output, labels)
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        len_samples = len(data)
        loss_meter.update(loss_value.detach().cpu().numpy(), len_samples)
        accu_value = (torch.argmax(output, dim=1) == labels).sum().detach().cpu().numpy() / len_samples
        accu_meter.update(accu_value, len_samples)
        print(
            f"[{current_epoch}/{total_epoches}],[{current_batch_count}/{len(dataloader)}] "
            f"Loss:{loss_meter.avg:.4f},Accu:{accu_meter.avg:.4f}")
        print(accu_meter.avg, loss_meter)


@torch.no_grad()
def val_model(model, dataloader, loss_func, current_epoch, total_epoches):
    model.eval()
    accu_meter = AverageMeter()
    loss_meter = AverageMeter()
    for current_batch_count, data in enumerate(dataloader):
        image, labels = data[0].to(device), data[1].to(device)
        output = model(image)
        predicted_labels = torch.argmax(output, dim=1)
        num_of_equal_labels = (predicted_labels == labels).sum().cpu().numpy()
        loss_value = loss_func(labels, output).detach().cpu().numpy()
        loss_meter.update(loss_value, len(data[0]))
        accu_meter.update(num_of_equal_labels / len(data[0]), len(data[0]))

        print(f"[{current_epoch}/{total_epoches}]] "
              f"Loss:{loss_meter.avg:.4f},Accu:{accu_meter.avg:.4f}")


device = "cuda" if torch.cuda.is_available() else "cpu"


def main(batch_size, epoches):
    train_dataset, test_dataset = get_dataset("mnist")
    len_validation = int(0.2 * len(train_dataset))
    len_train = len(train_dataset) - len_validation
    train_dataset, val_dataset = random_split(train_dataset,
                                              [len_train, len_validation])

    train_dataloader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet(num_classes=10).to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    # TODO:change optimizer
    # lr is learning rate
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum = )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("\n")
    for epoch in range(1, epoches + 1):
        train_model_epoch(model, train_dataloader, loss_func, optimizer, epoch,
                          epoches)
        val_model(model, validation_dataloader, loss_func, epoch, epoches)
    val_model(model, test_dataloader, loss_func, -1, -1)


if __name__ == "__main__":
    main(batch_size=256, epoches=10)
    pass
