import time
import numpy as np
import timm
import torch
from torch import nn
import lightning as L
from torchmetrics.functional import accuracy


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 512, normalize=False),
            # *block(128, 1024),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img

class FoolImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = Generator(1000, (3, 224, 224))
        
        # Load and freeze the classifier weights
        self.clf = timm.create_model('resnet18', pretrained=True)
        for param in self.clf.parameters():
            param.requires_grad = False
        self.clf.eval()  # Set to evaluation mode
        
    def forward(self, x):
        fool_image = self.generator(x)
        
        pred_y = self.clf(fool_image)
        return pred_y


class LitFoolImage(L.LightningModule):
    def __init__(self,learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = FoolImage()
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_y = self.model(x)
            
        loss = nn.CrossEntropyLoss()
        loss = loss(pred_y,y)
        
        with torch.no_grad():
            pred_y = pred_y.argmax(dim=1)
            y = y.argmax(dim=1)
            acc = accuracy(pred_y, y, task="multiclass", num_classes=1000)

            
        self.log("Loss", loss)
        self.log("Accuracy", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            pred_y = self.model(x)
            loss = nn.CrossEntropyLoss()
            loss = loss(pred_y,y)

            pred_y = pred_y.argmax(dim=1)
            y = y.argmax(dim=1)
            acc = accuracy(pred_y, y, task="multiclass", num_classes=1000)


        self.log("Val_Loss", loss)
        self.log("Val_Accuracy", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)
        return {
                "optimizer": optimizer,
            }
