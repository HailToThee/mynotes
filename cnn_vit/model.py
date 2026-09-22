import random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from cvt_small import CvT
def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class PatchCNN(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128 ,256, 3,padding=1),
            nn.BatchNorm2d(256),nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
            )
        self.proj=nn.Linear(256, dim)

    def forward(self,x):
        return self.proj(self.net(x).flatten(1))

class CNNViT(nn.Module):
    def __init__(self,img = 128, patch=32, classes=10, dim=256, heads=8, layers=4, drop=0.1):
        super().__init__()
        assert img%patch == 0
        self.patch = patch
        self.n = (img//patch)**2

        self.cnn = PatchCNN(dim)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1,self.n+1,dim))

        enc = nn.TransformerEncoderLayer(dim,heads, dim*4, drop, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc,layers)

        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim,classes)

        nn.init.trunc_normal_(self.cls,std=.02)
        nn.init.trunc_normal_(self.pos,std=.02)

    def forward(self,x):
        B,C,H,W=x.shape; P=self.patch
        x = x.unfold(2, P, P).unfold(3, P, P)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(B*self.n, C, P, P)
        x = self.cnn(x).view(B,self.n, -1)
        x = torch.cat((self.cls.expand(B,-1,-1),x),1)+self.pos
        x = self.transformer(x)
        return self.fc(self.norm(x[:,0]))

def run(model,loader,loss_fn,device,opt=None):
    model.train(opt is not None)
    loss_sum=correct=total=0
    for x,y in loader:
        x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
        with torch.set_grad_enabled(opt is not None):
            out=model(x); loss=loss_fn(out,y)
            if opt:
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss_sum+=loss.item()*x.size(0)
        correct+=(out.argmax(1)==y).sum().item()
        total+=x.size(0)
    return loss_sum/total,correct/total

def main():
    seed_all()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img,patch,batch,epochs=128,32,128,30
    mean=(.4914,.4822,.4465); std=(.247,.2435,.2616)

    train_tf=transforms.Compose([
        transforms.Resize((img,img)),transforms.RandomCrop(img,padding=8),
        transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean,std)])
    test_tf=transforms.Compose([
        transforms.Resize((img,img)),transforms.ToTensor(),transforms.Normalize(mean,std)])

    train_ds=datasets.CIFAR10("./data",train=True,download=True,transform=train_tf)
    test_ds=datasets.CIFAR10("./data",train=False,download=True,transform=test_tf)
    train_dl=DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers=8,pin_memory=True)
    test_dl=DataLoader(test_ds,batch_size=batch,shuffle=False,num_workers=8,pin_memory=True)

    # model=CNNViT(img,patch).to(device)
    model = CvT(classes=10).to(device)
    loss_fn=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)

    # print(f"device={device} params={sum(p.numel() for p in model.parameters()):,} patches={model.n}")
    best=0
    for e in range(1,epochs+1):
        tr_l,tr_a=run(model,train_dl,loss_fn,device,opt)
        with torch.no_grad(): te_l,te_a=run(model,test_dl,loss_fn,device)
        scheduler.step()
        print(f"{e:02d}/{epochs} | train {tr_l:.4f} {tr_a*100:.2f}% | test {te_l:.4f} {te_a*100:.2f}%")
        if te_a>best:
            best=te_a
            torch.save(model.state_dict(),"cvt_small.pt")

    print(f"Best: {best*100:.2f}%")

if __name__=="__main__":
    main()