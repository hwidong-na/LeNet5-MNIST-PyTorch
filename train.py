from model import Model
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

RGB = 3

class MNIST(mnist.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, n_colors=2):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)
        self.n_colors = n_colors

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img_ = torch.zeros(img.shape)
        img_ = img_.repeat((RGB,1,1))
        img_[target % self.n_colors,:,:] = img

        return img_, target



if __name__ == '__main__':
    batch_size = 256
    train_dataset = MNIST(root='./train', train=True, transform=ToTensor())
    test_dataset = MNIST(root='./test', train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = Model().cuda()
    sgd = SGD(model.parameters(), lr=1e-1)
    cross_error = CrossEntropyLoss().cuda()
    epoch = 100

    for _epoch in range(epoch):
        for idx, (train_x, train_label) in enumerate(train_loader):
            sgd.zero_grad()
            predict_y = model(train_x.float().cuda())
            _error = cross_error(predict_y, train_label.long().cuda())
            if idx % 10 == 0:
                print('idx: {}, _error: {}'.format(idx, _error))
            _error.backward()
            sgd.step()

        correct = 0
        _sum = 0

        for idx, (test_x, test_label) in enumerate(test_loader):
            predict_y = model(test_x.float().cuda()).detach().cpu()
            predict_ys = np.argmax(predict_y, axis=-1)
            _ = predict_ys == test_label
            correct += np.sum(_.numpy(), axis=-1)
            _sum += _.shape[0]

        print('epoch {}, accuracy: {:.2f}'.format(_epoch, correct / _sum))
        torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))
