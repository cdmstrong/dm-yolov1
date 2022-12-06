import torch
from model import *
from utils.dataset import *
from torch.utils.data import DataLoader
import torch.functional as F
def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss()(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss
class Train():
    def __init__(self, loss, epochs = 100, batch_size = 4, learn_rate = 0.01) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataset = dataSet()
        self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = Unet().to(self.device)
        self.loss = loss
        self.weights =  torch.ones([2]).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), learn_rate)
    def train(self):
        for epoch in range(self.epochs):
            for i,(img, label) in enumerate(self.data_loader):
                self.model.train()
                img = img.to(self.device)
                label = label.to(self.device)
                pred = self.model(img)
                loss = Focal_Loss(pred, label, self.weights, 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(epoch,self.epochs,i,len(self.dataset)//self.batch_size,loss))
            # if (epoch+1)%10==0:
            torch.save(self.model,"./unet"+str(epoch+1)+".pkl")
                
if __name__ == '__main__':
    train = Train(torch.nn.CrossEntropyLoss()).train()