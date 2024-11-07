import os

import wandb
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Timer

from data import get_dataloader
from model import LitFoolImage



def setup_training(log=False,save_dir='model_checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_dataloader(categories=1000,bath_size=250,train=True)
    valid_loader = get_dataloader(categories=1000,batch_size=250,train=False)

    model = LitFoolImage()
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename='fool_image_model',
            save_last=True,  # Save last model
            save_on_train_epoch_end=True  # Save at epoch end
        ),
        Timer(duration="00:04:50:00")
    ]
    logger = None
    if log:
        key = os.getenv("WANDB_API_KEY")
        wandb.login(key=key)
        logger = WandbLogger(project="Fool Image", name="1000 Classes")
    
    
    trainer = L.Trainer(
        max_epochs=10_000,
        check_val_every_n_epoch=100,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=4
    )

    return model, trainer, train_loader, valid_loader

def main(ckpt_path=None,log=False):
    model, trainer, train_loader, valid_loader = setup_training(log=log)
    trainer.fit(model, train_loader, valid_loader,ckpt_path=ckpt_path)

if __name__ == "__main__":
    main(log=True)

