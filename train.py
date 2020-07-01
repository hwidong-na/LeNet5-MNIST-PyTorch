import torch
import random
from model import Model
from model import SelfTaughtModel
from model import MixupModel
from model import SelfTaughtMixupModel
from ColoredMNIST import MNIST
from DataLoader import DataLoader
from DataLoader import UnlabeledDataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from operator import itemgetter

CLS = 10
def draw(loader, prefix, n=2, interval=100):
    fig = plt.figure(figsize=(n, CLS))
    for idx, (X, Y) in enumerate(loader):
        if idx % interval == 0:
            for c, (X_, y_) in enumerate(sorted(zip(X, Y), key=itemgetter(1))):
                if c >= CLS:
                    break
                for k in range(n):
                    plt.subplot(n,CLS,k*CLS+c+1) # one-based index
                    plt.imshow(X_[k].permute([1,2,0]), interpolation='none')
                    if k == 0:
                        plt.title("CLS: {}".format(c))
                    plt.xticks([])
                    plt.yticks([])
            # fig.tight_layout()
            plt.savefig("{}-{}.png".format(prefix,idx), dpi=fig.dpi)
            plt.close()
    
if __name__ == '__main__':
    batch_size = 10 #256
    n_s = 5
    n_q = 5
    test_n_s = 5
    test_n_q = 5
    train_ind = MNIST(root='./train', train=True, transform=ToTensor(), color_prob=0.9)
    train_ood = MNIST(root='./train', train=True, transform=ToTensor(), color_prob=0.5, ood=True, ulb=True)
    test_ind = MNIST(root='./test', train=False, transform=ToTensor(), color_prob=0.9) # ind colors same as training
    test_ood = MNIST(root='./test', train=False, transform=ToTensor(), color_prob=0.5, ood=True) # ood colors
    test_unk = MNIST(root='./test', train=False, transform=ToTensor(), n_colors=3, color_prob=0.5) # there are an unknown color
    train_spl_per_cls = 6000
    test_spl_per_cls = 1000
    train_loader = DataLoader(train_ind, batch_size, max_spl_per_cls=train_spl_per_cls, nDataLoaderThread=1, gSize=n_s+n_q, maxQueueSize=500)
    uloader = UnlabeledDataLoader(train_ood, batch_size*4, max_spl_per_cls=train_spl_per_cls, nDataLoaderThread=1, gSize=n_s+n_q, maxQueueSize=500)
    ind_loader = DataLoader(test_ind, batch_size, max_spl_per_cls=test_spl_per_cls, nDataLoaderThread=1, gSize=test_n_s+test_n_q, maxQueueSize=500)
    ood_loader = DataLoader(test_ood, batch_size, max_spl_per_cls=test_spl_per_cls, nDataLoaderThread=1, gSize=test_n_s+test_n_q, maxQueueSize=500)
    unk_loader = DataLoader(test_unk, batch_size, max_spl_per_cls=test_spl_per_cls, nDataLoaderThread=1, gSize=test_n_s+test_n_q, maxQueueSize=500)
    draw(train_loader, "train", n=n_s+n_q, interval=train_spl_per_cls)
    draw(uloader, "udata", n=n_s+n_q, interval=train_spl_per_cls)
    draw(ind_loader, "ind", n=test_n_s+test_n_q, interval=test_spl_per_cls)
    draw(ood_loader, "ood", n=test_n_s+test_n_q, interval=test_spl_per_cls)
    draw(unk_loader, "unk", n=test_n_s+test_n_q, interval=test_spl_per_cls)

    def do_train(model):
        lr_decay = 0.99
        epoch = 50
        interval = 5

        for _epoch in range(1,epoch+1):
            prec = model.fit(loader=train_loader, n_s=n_s, n_q=n_q).item()
            print('TRAIN epoch {}, accuracy: {:.2f}'.format(_epoch, prec))
            clr = model.updateLearningRate(lr_decay) 

            if _epoch % interval == 0:
                print('='*80)
                prec = model.infer(loader=ind_loader, n_s=test_n_s, n_q=test_n_q).item()
                print('IND epoch {}, accuracy: {:.2f}'.format(_epoch, prec))

                prec = model.infer(loader=ood_loader, n_s=test_n_s, n_q=test_n_q).item()
                print('OOD epoch {}, accuracy: {:.2f}'.format(_epoch, prec))

                prec = model.infer(loader=unk_loader, n_s=test_n_s, n_q=test_n_q).item()
                print('UNK epoch {}, accuracy: {:.2f}'.format(_epoch, prec))
                print('='*80)

    seed=random.randrange(10000)
    print("Random seed", seed)
    ### supervised learning
    random.seed(seed)
    torch.manual_seed(seed)
    sm = Model().cuda()
    do_train(sm)

    ### self-taught learning
    random.seed(seed)
    torch.manual_seed(seed)
    stm = SelfTaughtModel(uloader, pre=20).cuda()
    do_train(stm)
    stm.terminate()

    ### supervised mixup learning
    random.seed(seed)
    torch.manual_seed(seed)
    smm = MixupModel(alpha=0.1).cuda()
    do_train(smm)

    ### self-taught mixup learning
    random.seed(seed)
    torch.manual_seed(seed)
    stmm = SelfTaughtMixupModel(uloader, pre=20, alpha=0.1).cuda()
    do_train(stmm)
    stmm.terminate()
