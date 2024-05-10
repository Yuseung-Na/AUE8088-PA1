"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch
import wandb

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

torch.set_float32_matmul_precision('medium')

model_list = ['MyAlex', 
              'MyAlexDeep',
              'MyAlexBN', 
              'MyAlexDeepBN',
              'MyAlexInception',
              'MyAlexResidual',
              'MyAlexSE',
              'MyAlexTotal'
]

optimizer_params = {'type': 'AdamW', 'lr': 0.001}
scheduler_params = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.1}

if __name__ == "__main__":
    
    for current_model in model_list:
        cfg.WANDB_NAME = f'{current_model}-B{cfg.BATCH_SIZE}-{optimizer_params["type"]}'
        cfg.WANDB_NAME += f'-{scheduler_params["type"]}{optimizer_params["lr"]:.1E}'    

        model = SimpleClassifier(
            model_name = current_model,
            num_classes = cfg.NUM_CLASSES,
            optimizer_params = optimizer_params,
            scheduler_params = scheduler_params,
        )

        datamodule = TinyImageNetDatasetModule(
            batch_size = cfg.BATCH_SIZE,
        )

        wandb.login(key = cfg.WANDB_KEY, relogin = True)
        wandb_logger = WandbLogger(
            project = cfg.WANDB_PROJECT,
            save_dir = cfg.WANDB_SAVE_DIR,
            entity = cfg.WANDB_ENTITY,
            name = cfg.WANDB_NAME,
        )

        trainer = Trainer(
            accelerator = cfg.ACCELERATOR,
            devices = cfg.DEVICES,
            precision = cfg.PRECISION_STR,
            max_epochs = cfg.NUM_EPOCHS,
            check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
            logger = wandb_logger,
            callbacks = [
                LearningRateMonitor(logging_interval='epoch'),
                ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
            ],
        )

        trainer.fit(model, datamodule=datamodule)
        trainer.validate(ckpt_path='best', datamodule=datamodule)
        
        wandb.finish()
