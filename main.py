import os
import argparse
import wandb
import lightning as L
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, Timer

from data import get_dataloader
from model import LitFoolImage


def setup_training(batch_size, categories, max_epochs, duration="00:04:50:00", log=False, save_dir='model_checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_dataloader(categories, batch_size, train=True)

    model = LitFoolImage()
    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename='fool_image_model',
            save_last=True,  # Save last model
            save_on_train_epoch_end=True  # Save at epoch end
        ),
        Timer(duration=duration)
    ]
    logger = None
    if log:
        key = os.getenv("WANDB_API_KEY")
        wandb.login(key=key)
        logger = WandbLogger(project="Fool Image", name="1000 Classes")
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=4
    )

    return model, trainer, train_loader

def main(batch_size, categories, max_epochs, duration, log, save_dir, ckpt_path=None):
    model, trainer, train_loader = setup_training(batch_size, categories, max_epochs, duration, log, save_dir)
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fool Image model.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--categories", type=int, required=True, help="Number of categories (classes)")
    parser.add_argument("--max_epochs", type=int, required=True, help="Maximum number of epochs")
    parser.add_argument("--duration", type=str, default="00:04:50:00", help="Training duration in format DD:HH:MM:SS")
    parser.add_argument("--log", action="store_true", help="Enable logging with WandB")
    parser.add_argument("--save_dir", type=str, default="model_checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to checkpoint for resuming training")

    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        categories=args.categories,
        max_epochs=args.max_epochs,
        duration=args.duration,
        log=args.log,
        save_dir=args.save_dir,
        ckpt_path=args.ckpt_path
    )

