import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from minist.data_set import MnistData


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # N C H W
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        # 相当于flatten
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':

    dataset_train = MnistData()
    dataloader = DataLoader(dataset=dataset_train, batch_size=1, shuffle=True, num_workers=2)

    net = Net()
    # 学习率
    learning_rate = 1e-4
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # 设置为训练模式
    net.train()
    for id, (train, label) in enumerate(dataloader, 0):
        y_pred = net(train)

        # 计算损失
        loss = criterion(y_pred, label)

        # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)
        optimizer.zero_grad()

        # 反向传播：根据模型的参数计算loss的梯度
        loss.backward()

        # 调用Optimizer的step函数使它所有参数更新
        optimizer.step()

        # 每100个batch打印一次
        if id % 100 == 0:
            print(f"当前训练的进程是:{id * 32}/60000 loss is {loss:.4f}")

    torch.save(net.state_dict(), "model.pth")
    print("成功保存模型...")
