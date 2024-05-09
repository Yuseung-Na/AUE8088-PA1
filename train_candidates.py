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

model_list = ['alexnet', 
              'densenet121', 
              'mnasnet0_5', 
              'mobilenet_v2',
              'resnet18',
              'resnext50_32x4d',
              'swin_v2_t',
              'efficientnet_v2_s',
]

optimizer_params_list = [
    {'type': 'SGD', 'lr': 0.1},
    {'type': 'SGD', 'lr': 0.01},
    {'type': 'SGD', 'lr': 0.005},
    {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9},
    {'type': 'SGD', 'lr': 0.005, 'momentum': 0.95},
    {'type': 'SGD', 'lr': 0.005, 'momentum': 0.999},
    {'type': 'Adagrad', 'lr': 0.01},
    {'type': 'Adam', 'lr': 0.001},
    {'type': 'AdamW', 'lr': 0.001},
    {'type': 'RMSprop', 'lr': 0.01},
]

optimizer_schedul_params_list = [
    {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9},
    {'type': 'Adam', 'lr': 0.001},
]

scheduler_params_list = [
    {'type': 'StepLR', 'step_size': 30, 'gamma': 0.1},
    {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.1},
    {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2},
    {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.5},
    {'type': 'ExponentialLR', 'gamma': 0.95},
    {'type': 'CosineAnnealingLR', 'T_max': 10},
    {'type': 'CyclicLR', 'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'mode': 'triangular'},
    {'type': 'CyclicLR', 'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'mode': 'triangular2'},  
    {'type': 'CyclicLR', 'base_lr': 0.001, 'max_lr': 0.01, 'step_size_up': 5, 'mode': 'exp_range'},
]

if __name__ == "__main__":
    
    for current_model in model_list:
        cfg.WANDB_NAME = f'{current_model}-B{cfg.BATCH_SIZE}-{cfg.OPTIMIZER_PARAMS["type"]}'
        cfg.WANDB_NAME += f'-{cfg.SCHEDULER_PARAMS["type"]}{cfg.OPTIMIZER_PARAMS["lr"]:.1E}'    

        model = SimpleClassifier(
            model_name = current_model,
            num_classes = cfg.NUM_CLASSES,
            optimizer_params = cfg.OPTIMIZER_PARAMS,
            scheduler_params = cfg.SCHEDULER_PARAMS,
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
    
    for current_optimizer_params in optimizer_params_list:
        cfg.WANDB_NAME = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{current_optimizer_params["type"]}'
        cfg.WANDB_NAME += f'-{cfg.SCHEDULER_PARAMS["type"]}{current_optimizer_params["lr"]:.1E}'    

        model = SimpleClassifier(
            model_name = cfg.MODEL_NAME,
            num_classes = cfg.NUM_CLASSES,
            optimizer_params = current_optimizer_params,
            scheduler_params = cfg.SCHEDULER_PARAMS,
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
        
    for current_optimizer_params in optimizer_schedul_params_list:    
        for current_scheduler_params in scheduler_params_list:
            cfg.WANDB_NAME = f'{cfg.MODEL_NAME}-B{cfg.BATCH_SIZE}-{current_optimizer_params["type"]}'
            cfg.WANDB_NAME += f'-{current_scheduler_params["type"]}{current_optimizer_params["lr"]:.1E}' 
            
            if current_optimizer_params["type"] == 'Adam' and current_scheduler_params["type"] == 'CyclicLR':
                continue

            model = SimpleClassifier(
                model_name = cfg.MODEL_NAME,
                num_classes = cfg.NUM_CLASSES,
                optimizer_params = current_optimizer_params,
                scheduler_params = current_scheduler_params,
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

