
from functools import partial
import torch
import torch.nn.functional as F
# import shutil
from torch.cuda.amp import autocast
import os
from torch.utils.data import DataLoader
import numpy as np
from glob import glob
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import argparse
from PIL import Image
import skimage.io
import cv2
import time
import albumentations as A
from  torchvision import utils as vutils
# from nvidia.dali.backend import TensorListCPU

import sys
sys.path.append('../')
from utils.config import parse
from EffUNet.model import UnetAux_v2,UnetAux_v2_WS, compare_model
# from submit_v24.model import UnetAux_v2
from EffUNet.model import UnetAux
from EffUNet.train import get_train_val_dfs, make_data_generator
from utils.models import model_dict, _load_model_weights


BATCH_SIZE = 16
INPUT_DIR = '/boot/data1/kang_data/zkxt21_seaice/trainData/image'


SAVE_DIR = '/code/lisijiang/zkxt_src/submit/EffUNet_stage5/20240118_182155/test_predict'

CHECKPOINT = '/code/lisijiang/zkxt_src/submit/EffUNet_stage5/20240118_182155 /checkpoints/20240118_182155_checkpoint.pth.tar'
config = parse('/code/lisijiang/zkxt_src/submit/EffUNet_stage5/20240118_182155 /20240118_182155.yaml')

train_df, val_df = get_train_val_dfs(config)
val_datagen = make_data_generator(config, val_df, stage='validate')


def main():

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    model = UnetAux_v2(encoder_name = 'timm-tf_efficientnet_lite3', encoder_weights=None, in_channels=3, classes=2, encoder_depth=4, decoder_attention_type='scse', decoder_channels=(128,64,32,16)).cuda()
    # model = UnetAux_v2_WS(encoder_name = 'timm-tf_efficientnet_lite3', encoder_weights=None, in_channels=3, classes=2, encoder_depth=4, decoder_attention_type='scse', decoder_channels=(128,64,32,16)).cuda()
    
    # [unet, Linknet, pspnet, fpn, deeplabv3, deeplabv3+]
    # model = compare_model(model_name='deeplabv3+').cuda()

    if os.path.isfile(CHECKPOINT):
        print("=> loading CHECKPOINT '{}'".format(CHECKPOINT))
        model = _load_model_weights(model, CHECKPOINT)

    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    softmax = partial(torch.softmax, dim=1)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_datagen,total = len(val_datagen),leave=True,ncols=50)):
            data = batch['image'].cuda()
            name = batch['name']

            m_logits = softmax(model(data)[0])
            # m_logits = softmax(model(data))

            outputs = torch.argmax(m_logits,dim=1)

            outputs = outputs.cpu().numpy()

            for j in range(outputs.shape[0]):
                save_img_pth  = SAVE_DIR + '/' + name[j]
                cv2.imwrite(save_img_pth, outputs[j]*255)


if __name__ == '__main__':

    main()
