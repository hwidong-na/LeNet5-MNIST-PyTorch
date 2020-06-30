from model import Model
from ColoredMNIST import MNIST
from DataLoader import DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def draw(loader, prefix, n=2, interval=100):
    fig = plt.figure(figsize=(n**2, n**2))
    for idx, (X, Y) in enumerate(loader):
        if idx % interval == 0:
            for c, (X_, y_) in enumerate(zip(X, Y)):
                for k in range(n):
                    plt.subplot(n*2,5,c*n+k+1) # one-based index
                    plt.imshow(X_[k].permute([1,2,0]), interpolation='none')
                    plt.title("CLS: {}".format(k, y_.item()))
                    plt.xticks([])
                    plt.yticks([])
            fig.tight_layout()
            plt.savefig("{}-{}.png".format(prefix,idx), dpi=fig.dpi)
            plt.close()
    
if __name__ == '__main__':
    batch_size = 10 #256
    n_s = 3
    n_q = 1
    test_n_s = 3
    test_n_q = 1
    train_ind = MNIST(root='./train', train=True, transform=ToTensor(), color_prob=0.9)
    test_ind = MNIST(root='./test', train=False, transform=ToTensor(), color_prob=0.9) # ind colors same as training
    test_ood = MNIST(root='./test', train=False, transform=ToTensor(), color_prob=0.5, ood=True) # ood colors
    test_unk = MNIST(root='./test', train=False, transform=ToTensor(), n_colors=3, color_prob=0.5) # there are an unknown color
    train_loader = DataLoader(train_ind, batch_size=batch_size, max_spl_per_cls=6000, nDataLoaderThread=5, gSize=n_s+n_q, maxQueueSize=500)
    ind_loader = DataLoader(test_ind, batch_size=batch_size, max_spl_per_cls=1000, nDataLoaderThread=5, gSize=test_n_s+test_n_q, maxQueueSize=500)
    ood_loader = DataLoader(test_ood, batch_size=batch_size, max_spl_per_cls=1000, nDataLoaderThread=5, gSize=test_n_s+test_n_q, maxQueueSize=500)
    unk_loader = DataLoader(test_unk, batch_size=batch_size, max_spl_per_cls=1000, nDataLoaderThread=5, gSize=test_n_s+test_n_q, maxQueueSize=500)
    draw(train_loader, "train", n=n_s+n_q, interval=100)
    draw(ind_loader, "ind", n=test_n_s+test_n_q, interval=50)
    draw(ood_loader, "ood", n=test_n_s+test_n_q, interval=50)
    draw(unk_loader, "unk", n=test_n_s+test_n_q, interval=50)

    model = Model().cuda()
    epoch = 50
    interval = 5

    for _epoch in range(1,epoch+1):
        prec = model.fit(loader=train_loader, n_s=n_s, n_q=n_q).item()
        print('TRAIN epoch {}, accuracy: {:.2f}'.format(_epoch, prec))

        if _epoch % interval == 0:
            print('='*80)
            prec = model.infer(loader=ind_loader, n_s=test_n_s, n_q=test_n_q).item()
            print('IND epoch {}, accuracy: {:.2f}'.format(_epoch, prec))

            prec = model.infer(loader=ood_loader, n_s=test_n_s, n_q=test_n_q).item()
            print('OOD epoch {}, accuracy: {:.2f}'.format(_epoch, prec))

            prec = model.infer(loader=unk_loader, n_s=test_n_s, n_q=test_n_q).item()
            print('UNK epoch {}, accuracy: {:.2f}'.format(_epoch, prec))
            print('='*80)
