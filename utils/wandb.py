import wandb
import os
import numpy as np


def init_wandb(cfg) -> None:
    """
    Initialize project on Weights & Biases
    Args:
        model (Torch Model): Model for Training
        args (TrainOptions,optional): TrainOptions class (refer options/train_options.py). Defaults to None.
    """
    
    wandb.init(
        name=cfg.WANDB.NAME,
        config=cfg,
        project=cfg.WANDB.PROJECT,
        resume="allow",
        id=cfg.WANDB.RESTORE_NAME
    )


def wandb_log(train_loss, lr, iter):
    """
    Logs the accuracy and loss to wandb
    Args:
        train_loss (float): Training loss
        val_loss (float): Validation loss
        train_acc (float): Training Accuracy
        val_acc (float): Validation Accuracy
        epoch (int): Epoch Number
    """

    wandb.log({
        'Loss': train_loss,
        'Learning Rate': lr,
    }, step=iter)

def wandb_log_seg(train_loss, train_loss_ce, train_loss_cons, lr, it):
    wandb.log({
        'Train Loss': train_loss,
        'Learning Rate': lr,
        'Train Iteration': it,
        'loss_ce': train_loss_ce,
        'loss_cons': train_loss_cons
    })

def wandb_log_NAL(loss,lr,it):
    wandb.log({
        'Total Loss':loss[0],
        'CE Loss':loss[1],
        'WCE Loss':loss[2],
        'Learning Rate': lr
    },step=it)