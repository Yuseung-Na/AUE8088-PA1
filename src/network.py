# Python packages
from termcolor import colored
from typing import Dict
import copy

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import nn
from torchvision import models
from torchvision.models.alexnet import AlexNet
import torch

# Custom packages
from src.metric import MyAccuracy
from src.metric import MyF1Score
import src.config as cfg
from src.util import show_setting

# MyNetworks
from src.my_network import MyAlex
from src.my_network import MyAlexDeep
from src.my_network import MyAlexBN
from src.my_network import MyAlexDeepBN
from src.my_network import MyAlexInception
from src.my_network import MyAlexResidual
from src.my_network import MyAlexSE
from src.my_network import MyAlexTotal

class SimpleClassifier(LightningModule):
    def __init__(self,
                 model_name: str = 'resnet18',
                 num_classes: int = 200,
                 optimizer_params: Dict = dict(),
                 scheduler_params: Dict = dict(),
        ):
        super().__init__()

        # Network
        if model_name == 'MyAlex':
            self.model = MyAlex(num_classes)
        elif model_name == 'MyAlexDeep':
            self.model = MyAlexDeep(num_classes)
        elif model_name == 'MyAlexBN':
            self.model = MyAlexBN(num_classes)
        elif model_name == 'MyAlexDeepBN':
            self.model = MyAlexDeepBN(num_classes)
        elif model_name == 'MyAlexInception':
            self.model = MyAlexInception(num_classes)
        elif model_name == 'MyAlexResidual':
            self.model = MyAlexResidual(num_classes)
        elif model_name == 'MyAlexSE':
            self.model = MyAlexSE(num_classes)
        elif model_name == 'MyAlexTotal':
            self.model = MyAlexTotal(num_classes)
        else:
            models_list = models.list_models()
            assert model_name in models_list, f'Unknown model name: {model_name}. Choose one from {", ".join(models_list)}'
            self.model = models.get_model(model_name, num_classes=num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Metric
        self.accuracy = MyAccuracy()
        self.f1_score = MyF1Score(num_classes=num_classes)

        # Hyperparameters
        self.save_hyperparameters()

    def on_train_start(self):
        show_setting(cfg)

    def configure_optimizers(self):
        optim_params = copy.deepcopy(self.hparams.optimizer_params)
        optim_type = optim_params.pop('type')
        optimizer = getattr(torch.optim, optim_type)(self.parameters(), **optim_params)

        scheduler_params = copy.deepcopy(self.hparams.scheduler_params)
        scheduler_type = scheduler_params.pop('type')
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)        
        self.log_dict({'loss/train': loss, 'accuracy/train': accuracy, 'f1_score/train': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self._common_step(batch)
        accuracy = self.accuracy(scores, y)
        f1_score = self.f1_score(scores, y)        
        self.log_dict({'loss/val': loss, 'accuracy/val': accuracy, 'f1_score/val': f1_score},
                      on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self._wandb_log_image(batch, batch_idx, scores, frequency = cfg.WANDB_IMG_LOG_FREQ)

    def _common_step(self, batch):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def _wandb_log_image(self, batch, batch_idx, preds, frequency = 100):
        if not isinstance(self.logger, WandbLogger):
            if batch_idx == 0:
                self.print(colored("Please use WandbLogger to log images.", color='blue', attrs=('bold',)))
            return

        if batch_idx % frequency == 0:
            x, y = batch
            preds = torch.argmax(preds, dim=1)
            self.logger.log_image(
                key=f'pred/val/batch{batch_idx:5d}_sample_0',
                images=[x[0].to('cpu')],
                caption=[f'GT: {y[0].item()}, Pred: {preds[0].item()}'])
