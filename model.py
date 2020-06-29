import sys
import time
import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        h = y
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y, h

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred  =  pred . t ()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res  = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion  = nn.CrossEntropyLoss()

    def forward(self, x, label, n_s=1, n_q=1):
        
        out_anchor      = torch.mean(x[:,:n_s,:],1)
        stepsize        = out_anchor.size()[0]
        out_positive    = x[:,n_s:n_s+n_q,:].reshape([stepsize*n_q, -1])
        
        pos = out_positive.unsqueeze(-1).expand(-1,-1,stepsize)
        anc = out_anchor.unsqueeze(-1).expand(-1,-1,stepsize*n_q).transpose(0,2)
        output      = -1 * (F.pairwise_distance(pos,anc)**2)
        label       = label.unsqueeze(-1).expand(-1,n_q).reshape([stepsize*n_q])
        nloss       = self.criterion(output, label)
        prec1       = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1, ))

        return nloss, prec1   

def draw(X, Y, prefix, n=2):
    fig = plt.figure(figsize=(n**2, n**2))
    for i, (X_, y_) in enumerate(zip(X, Y)):
        for k in range(n):
            plt.subplot(n*2,5,i*n+1+k)
            plt.imshow(X_[k].permute([1,2,0]), interpolation='none')
            plt.title("CLS: {}".format(k, y_.item()))
            plt.xticks([])
            plt.yticks([])
    plt.savefig("{}.png".format(prefix), dpi=fig.dpi)
    plt.close()
    
class Model(nn.Module):
    def __init__(self, n_s=1, n_q=1):
        super(Model, self).__init__()

        self.loss = Loss()
        self.net = Net()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        return self.net(x)
    
    def compute_loss(self, x, y, n_s, n_q):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        if next(self.parameters()).is_cuda:
            x = x.cuda()
            y = y.cuda()
        s = x.shape
        x = x.reshape([s[0]*s[1]]+list(s[2:]))
        x, h = self.net(x)
        x = x.reshape([s[0], s[1]]+list(x.shape[1:]))
        l, p = self.loss(x, y, n_s, n_q)
        return l, p
        
    def fit(self, loader, n_s, n_q, prefix="train"):
        
        self.train();
        
        loss = 0
        top1 = 0
        interval = 100
                
        # it is same for every batch
        y_ = [c for c in range(loader.batch_size)]
        for idx, (x, y) in enumerate(loader, 1):
            # if idx % interval == 0:
            #     draw(x,y,"{}.{}".format(prefix, idx))
            tstart = time.time()

            l, p = self.compute_loss(x, y_, n_s, n_q)
            
            self.zero_grad()
            l.backward()
            self.optimizer.step()
            
            loss += l.detach().cpu()
            top1 += p[0]

            telapsed = time.time() - tstart
            
            sys.stdout.write("\rProcessing (%d/%d) "%(idx*loader.batch_size, loader.nFiles));
            sys.stdout.write("Loss %f EER/T1 %2.3f%% - %.2f Hz "%(loss/idx, top1/idx, loader.batch_size/telapsed));
            sys.stdout.write("Q:(%d/%d)"%(loader.qsize(), loader.maxQueueSize));
            sys.stdout.flush();

        sys.stdout.write("\n");

        return top1 / idx

    def infer(self, loader, n_s, n_q, prefix="ind"):
        self.eval();
        
        loss = 0
        top1 = 0
        interval = 100
                
        # it is same for every batch
        y_ = [c for c in range(loader.batch_size)]
        for idx, (x, y) in enumerate(loader, 1):
            # if idx % interval == 0:
            #     draw(x,y,"{}.{}".format(prefix, idx))
            tstart = time.time()

            l, p = self.compute_loss(x, y_, n_s, n_q)
            
            loss += l.detach().cpu()
            top1 += p[0]

            telapsed = time.time() - tstart
            
            sys.stdout.write("\rProcessing (%d/%d) "%(idx*loader.batch_size, loader.nFiles));
            sys.stdout.write("Loss %f EER/T1 %2.3f%% - %.2f Hz "%(loss/idx, top1/idx, loader.batch_size/telapsed));
            sys.stdout.write("Q:(%d/%d)"%(loader.qsize(), loader.maxQueueSize));
            sys.stdout.flush();

        sys.stdout.write("\n");

        return top1 / idx

