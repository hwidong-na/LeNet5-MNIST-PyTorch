from model import Model
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import mnist
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from torch.utils.data import DataLoader
from DataLoader import DataLoader
from torchvision.transforms import ToTensor
import random

RGB = 3

def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def collapse_labels(labels, n_classes):
    """Collapse 10 classes into n_classes classes."""
    assert n_classes in [2, 3, 5, 10]
    bin_width = 10 // n_classes
    #return (labels / bin_width).clamp(max=n_classes - 1)
    return min(int(labels / bin_width), n_classes - 1)

def corrupt(labels, n_classes, prob):
    """Corrupt a fraction of labels by shifting it + o (mod n_classes),
    according to bernoulli(prob), where o is a random offset

    Generalizes torch_xor's role of label flipping for the binary case.
    """
    #keep = torch_bernoulli(prob, len(labels)).bool()
    if random.random() < prob:
        return labels
    offset = random.randrange(1, n_classes)
    #return torch.where(keep, labels, (labels + offset) % n_classes)
    return (labels + offset) % n_classes

class MNIST(mnist.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, n_colors=2, n_classes=10, color_prob=1., label_prob=1.):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)
        self.n_colors = n_colors
        self.n_classes = n_classes
        self.color_prob = color_prob
        self.label_prob = label_prob

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

        color = collapse_labels(target, self.n_colors)
        color = corrupt(color, self.n_colors, self.color_prob)
        label = collapse_labels(target, self.n_classes)
        label = corrupt(label, self.n_classes, self.label_prob)
        img_ = torch.zeros(img.shape)
        img_ = img_.repeat((RGB,1,1))
        img_[color,:,:] = img

        return img_, label

if __name__ == '__main__':
    batch_size = 10 #256
    n_s = 1
    n_q = 1
    test_n_s = 1
    test_n_q = 1
    train_dataset = MNIST(root='./train', train=True, transform=ToTensor(), color_prob=0.9)
    test_dataset = MNIST(root='./test', train=False, transform=ToTensor(), color_prob=0.1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, max_spl_per_cls=6000, nDataLoaderThread=5, gSize=n_s+n_q, maxQueueSize=500)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, max_spl_per_cls=1000, nDataLoaderThread=5, gSize=test_n_s+test_n_q, maxQueueSize=500)
    model = Model().cuda()
    epoch = 100
    interval = 10

    for _epoch in range(1,epoch+1):
        prec = model.fit(loader=train_loader, n_s=n_s, n_q=n_q).item()
        print('TRAIN epoch {}, accuracy: {:.2f}'.format(_epoch, prec))

        if _epoch % interval == 0:
            prec = model.infer(loader=test_loader, n_s=test_n_s, n_q=test_n_q).item()
            print('TEST accuracy: {:.2f}'.format(_epoch, prec))
        # correct = 0
        # _sum = 0

        # for idx, (test_x, test_label) in enumerate(test_loader):
        #     predict_y = model(test_x.float().cuda()).detach().cpu()
        #     predict_ys = np.argmax(predict_y, axis=-1)
        #     _ = predict_ys == test_label
        #     correct += np.sum(_.numpy(), axis=-1)
        #     _sum += _.shape[0]

        # print('epoch {}, accuracy: {:.2f}'.format(_epoch, correct / _sum))
        # torch.save(model, 'models/mnist_{:.2f}.pkl'.format(correct / _sum))
