import numpy as np
import torch
import torch.utils.data as dt


# 加载Minst数据集
def load_data(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()


class MnistData(dt.Dataset):

    def __init__(self, train=True) -> None:
        self.train = train
        if self.train:
            self.train_data = torch.from_numpy(x_train).view(x_train.shape[0], 1, x_train.shape[1],
                                                             x_train.shape[2]).to(dtype=torch.float32)
            self.train_label = torch.from_numpy(y_train)
        else:
            self.test_data = torch.from_numpy(x_test).view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).to(
                dtype=torch.float32)
            self.test_label = torch.from_numpy(y_test)

    def __len__(self) -> int:
        if self.train:
            return self.train_data.size()[0]
        else:
            return self.test_data.size()[0]

    def __getitem__(self, index: int):
        if self.train:
            return self.train_data[index].view(-1, 28, 28), self.train_label[index].to(dtype=torch.int64)
        else:
            return self.test_data[index].view(-1, 28, 28), self.test_label[index].to(dtype=torch.int64)
