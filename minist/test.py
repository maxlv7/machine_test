import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from minist.data_set import MnistData
from minist.train import Net

dataset_train = MnistData()
data_loader = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, num_workers=1)

net = Net()
net.load_state_dict(torch.load('model.pth'))
net.eval()

test_loss = 0
correct = 0
if __name__ == '__main__':

    with torch.no_grad():
        for data, target in data_loader:
            predict = net(data)
            test_loss += F.nll_loss(predict, target, reduction='sum').item()  # sum up batch loss
            pred = predict.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
