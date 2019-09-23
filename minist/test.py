import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from minist.data_set import MnistData
from minist.train import Net

dataset_test = MnistData()
data_loader = DataLoader(dataset=dataset_test, batch_size=32, shuffle=True, num_workers=1)

net = Net()

# 加载状态字典(实际上是给网络加载参数)
net.load_state_dict(torch.load('model.pth'))
# 设置 dropout 和 batch normalization 为评估(去掉bn，dropout层)
net.eval()

test_loss = 0
correct = 0
if __name__ == '__main__':

    # 不追综梯度变化
    with torch.no_grad():
        for data, target in data_loader:
            predict = net(data)
            test_loss += F.nll_loss(predict, target, reduction='sum').item()  # sum up batch loss
            pred = predict.argmax(dim=1)  # get the index of the max log-probability
            correct = correct + pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)

    print('\n测试数据集: 平均损失为: {:.4f}, 正确率为: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
