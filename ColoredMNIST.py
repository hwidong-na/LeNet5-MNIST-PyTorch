import random
import torch
from torchvision.datasets import mnist
from PIL import Image

RGB = 3

def torch_bernoulli(p, size):
    return (torch.rand(size) < p).float()

def collapse_labels(labels, n_classes):
    """Collapse 10 classes into n_classes classes."""
    assert n_classes in [2, 3, 5, 10]
    bin_width = 10 // n_classes
    return min(int(labels / bin_width), n_classes - 1)

def corrupt(labels, n_classes, prob):
    """Corrupt a fraction of labels by shifting it + o (mod n_classes),
    according to bernoulli(prob), where o is a random offset

    Generalizes torch_xor's role of label flipping for the binary case.
    """
    if random.random() < prob:
        return labels
    offset = random.randrange(1, n_classes)
    return (labels + offset) % n_classes

class MNIST(mnist.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, n_colors=2, n_classes=10, color_prob=1., label_prob=1., ood=False):
        super(MNIST, self).__init__(root, train, transform, target_transform, download)
        self.n_colors = n_colors
        self.n_classes = n_classes
        self.color_prob = color_prob
        self.label_prob = label_prob
        self.ood = ood

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
        if self.ood: # set unknown color
            img_[RGB-1,:,:] = img

        return img_, label

