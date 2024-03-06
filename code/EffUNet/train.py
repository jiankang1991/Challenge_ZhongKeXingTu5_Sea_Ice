import os
import torch
from torch import nn
import torch.nn.functional as F
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from tensorboardX import SummaryWriter
import sys
sys.path.append("../")
from utils.models import _load_model_weights, model_dict, _load_model_weights_v2
from utils.dataGen import make_data_generator
from utils.losses import get_loss
from utils.metrics import Evaluator, MetricTracker
from utils.lr_scheduler import LR_Scheduler

from model import UnetAux, UnetAux_v2, UnetAux_v3, UnetAux_v4, EarlyStopping, UnetAux_v2_WS, compare_model

class Trainer:
    """Object for training `solaris` models using PyTorch. """
    def __init__(self, config, custom_losses=None):
        
        self.sv_name = config['sv_name']
        self.checkpoint_dir = config['checkpoint_dir']
        self.logs_dir = config['logs_dir']
        self.config = config
        self.batch_size = self.config['batch_size']
        self.model_name = self.config['model_name']
        self.model_path = self.config.get('model_path', None)
        # self.num_classes = self.config['data_specs']['num_classes']

        # self.model = UnetAux(**self.config['model_specs'])
        self.model = UnetAux_v2(**self.config['model_specs'])
        # self.model = compare_model(**self.config['compare_model'])
        # self.model = UnetAux_v2_WS(**self.config['model_specs'])
        # self.model = UnetAux_v4(**self.config['model_specs'])
        # self.es = EarlyStopping(mode='max', patience=5)
        if self.model_path:
            # self.model = _load_model_weights(self.model, self.model_path)
            self.model = _load_model_weights_v2(self.model, self.model_path)

        self.train_df, self.val_df = get_train_val_dfs(self.config)
        self.train_datagen = make_data_generator(self.config,
                                                 self.train_df, stage='train')
        self.val_datagen = make_data_generator(self.config,
                                               self.val_df, stage='validate')
        self.epochs = self.config['training']['epochs']
        self.lr = self.config['training']['lr']
        self.loss = get_loss(self.config['training'].get('loss'),
                             self.config['training'].get('loss_weights'),
                             custom_losses)

        self.metrics = Evaluator(self.config['model_specs']['classes'])
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_count = torch.cuda.device_count()
        else:
            self.gpu_count = 0

        self.train_writer = SummaryWriter(os.path.join(self.logs_dir, 'runs', self.sv_name, 'training'))
        self.val_writer = SummaryWriter(os.path.join(self.logs_dir, 'runs', self.sv_name, 'val'))

        self.initialize_model()

    def initialize_model(self):
        if self.gpu_available:
            self.model = self.model.cuda()
            if self.gpu_count > 1:
                self.model = torch.nn.DataParallel(self.model)        
        # self.optimizer = torch.optim.SGD(
        #             self.model.parameters(), lr=self.lr,
        #             momentum=0.9, weight_decay=1e-4, nesterov=True
        #         )
        self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.lr
                )
        self.lr_scheduler = LR_Scheduler('poly', self.lr, self.epochs + 1, len(self.train_datagen))

    def run(self):
        """
        the main function to run
        """
        best_metric = 0
        for epoch in range(1, self.epochs+1):
            print('Epoch {}/{}'.format(epoch, self.epochs))
            print('-' * 10)
            self.train(epoch, best_metric)
            metric_v = self.val(epoch)

            # if self.es.step(torch.tensor(metric_v)):
            #     print("early stopping is triggered")
            #     break  # early stop criterion is met, we can stop now

            is_best_metric = metric_v > best_metric
            best_metric = max(metric_v, best_metric)

            self.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
                'best_metric': best_metric,
                'optimizer': self.optimizer.state_dict()
            }, None)

    def train(self, epoch, best_metric):
        """Run training on the model."""
        Losses = MetricTracker()
        self.model.train()
        for idx, batch in enumerate(tqdm(self.train_datagen, desc="training", ascii=True, ncols=60)):
            if torch.cuda.is_available():
                data = batch['image'].cuda()
                target = batch['mask'].cuda().long()
                if self.config['model_specs'].get('aux_params', None) is not None:
                    label = batch['label'].cuda().long()
            self.optimizer.zero_grad()
            seg0, seg2, seg3 = self.model(data)
            # seg0 = self.model(data)

            with torch.no_grad():
                targets_down2 = F.interpolate(target.unsqueeze(1).float(), size=seg2.shape[2:], mode='nearest')
                targets_down3 = F.interpolate(target.unsqueeze(1).float(), size=seg3.shape[2:], mode='nearest')

            seg0_loss = self.loss(seg0, target)
            seg2_loss = self.loss(seg2, targets_down2.squeeze().long())
            seg3_loss = self.loss(seg3, targets_down3.squeeze().long())
            loss = seg0_loss + 0.5 * (seg2_loss + seg3_loss)
            
            # loss = self.loss(seg0, target)
            
            loss.backward()
            self.optimizer.step()

            Losses.update(loss.item(), data.size(0))

            self.lr_scheduler(self.optimizer, idx, epoch, best_metric)

        info = {
                "Loss": Losses.avg,
        }
        for tag, value in info.items():
            self.train_writer.add_scalar(tag, value, epoch)
        
        print('Train Loss: {:.6f}'.format(
                Losses.avg
                ))

        return None

    def val(self, epoch):
        self.model.eval()
        torch.cuda.empty_cache()
        
        # val_Metric = MetricTracker()
        self.metrics.reset()
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.val_datagen, desc="val", ascii=True, ncols=60)):
                if torch.cuda.is_available():
                    data = batch['image'].cuda()
                    target = batch['mask'].cpu()

                logits = self.model(data)
                outputs = torch.argmax(logits[0], dim=1)
                # outputs = torch.argmax(logits, dim=1)
                outputs = outputs.cpu().numpy()
                target = target.numpy()
                # for j in range(outputs.shape[0]):
                #     cv2.imwrite('/code/lisijiang/zkxt_src/EffUNet/gt'+ '/' + str(j) + '.png',outputs[j]*255)
                # val_Metric.update(self.metrics(outputs, target), outputs.size(0))
                self.metrics.add_batch(target, outputs)
        fwIoU = self.metrics.Frequency_Weighted_Intersection_over_Union()
        info = {
            "FwIoU": fwIoU
        }
        for tag, value in info.items():
            self.val_writer.add_scalar(tag, value, epoch)
        
        print('Val FwIoU: {:.6f}'.format(
                fwIoU
                ))

        return fwIoU

    def save_checkpoint(self, state, is_best):
        filename = os.path.join(self.checkpoint_dir, self.sv_name + '_checkpoint.pth.tar')
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(self.checkpoint_dir, self.sv_name + '_model_best.pth.tar'))

def get_train_val_dfs(config):
    """Get the training and validation dfs based on the contents of ``config``.

    This function uses the logic described in the documentation for the config
    files to determine where to find training and validation dataset files.
    See the docs and the comments in solaris/data/config_skeleton.yml for
    details.

    Arguments
    ---------
    config : dict
        The loaded configuration dict for model training and/or inference.

    Returns
    -------
    train_df, val_df : :class:`tuple` of :class:`dict` s
        :class:`dict` s containing two columns: ``'image'`` and ``'label'``.
        Each column corresponds to paths to find matching image and label files
        for training.
    """

    train_df = pd.read_csv(config['training_data_csv'])

    if config['data_specs']['val_holdout_frac'] is None:
        if config['validation_data_csv'] is None:
            raise ValueError(
                "If val_holdout_frac isn't specified in config,"
                " validation_data_csv must be.")
        val_df = pd.read_csv(config['validation_data_csv'])
    
    else:
        val_frac = config['data_specs']['val_holdout_frac']
        val_subset = np.random.choice(train_df.index,
                                      int(len(train_df)*val_frac),
                                      replace=False)
        val_df = train_df.loc[val_subset]
        # remove the validation samples from the training df
        train_df = train_df.drop(index=val_subset)

    return train_df, val_df

