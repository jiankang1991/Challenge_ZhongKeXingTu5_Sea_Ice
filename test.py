
from functools import partial
import torch
# import shutil
from torch.cuda.amp import autocast
import os

import numpy as np
from glob import glob
from collections import defaultdict, OrderedDict
from tqdm import tqdm

import argparse
from PIL import Image
import skimage.io
import cv2
from concurrent.futures import ThreadPoolExecutor
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy

from model import _load_model_weights, UnetAux_v2
# BATCH_SIZE = 8
BATCH_SIZE = 12
INPUT_DIR = '/input_path'
# VAL_CSV = '../data/val.csv'
SAVE_DIR = '/output_path'
CHECKPOINT = '20230313_142125_checkpoint_state.pth.tar'
# VAL_IMG_NM = pd.read_csv(VAL_CSV)['image'].tolist()
threadpool = ThreadPoolExecutor(max_workers=4)

def save_img(SAVE_DIR,img_nm,cls_map):
    save_img_pth = os.path.join(SAVE_DIR, img_nm + '.png')
    Image.fromarray(cls_map).save(save_img_pth)
    
def GetImgSize(img_pth):
    if img_pth.endswith('.png'):
        im = Image.open(img_pth)
        w, h = im.size # the order is different
        return (h, w, 3), img_pth
    else:
        im = skimage.io.imread(img_pth)
        return im.shape, img_pth

@pipeline_def
def simple_pipeline(img_pths):
    imgs, idxs = fn.readers.file(files=img_pths, name="Reader")
    imgs = fn.decoders.image(imgs, device='cpu')
    imgs = fn.normalize(imgs.gpu(), mean=255*0.5, stddev=255*0.125)
    return imgs, idxs

class TestImages:
    """
    get the image dataloaders of test_img dir
    """
    def __init__(self, img_dir):
        self.img_pths = glob(os.path.join(img_dir, '*'))
        # self.img_pths = list(map(lambda x: os.path.join(img_dir, x), VAL_IMG_NM))
        self.img_cluster = defaultdict()

        for img_sz, img_pth in map(GetImgSize, self.img_pths):
            # print(img_sz, img_pth)
            if img_sz not in self.img_cluster:
                self.img_cluster[img_sz] = [img_pth]
            else:
                self.img_cluster[img_sz].append(img_pth)

def main():
    
    model = UnetAux_v2(encoder_name = 'timm-tf_efficientnet_lite3', encoder_weights=None, in_channels=3, classes=2, encoder_depth=4, decoder_attention_type='scse', decoder_channels=(128,64,32,16)).cuda()
    
    
    model.load_state_dict(torch.load(CHECKPOINT))
    model.eval()
    
    testData = TestImages(img_dir=INPUT_DIR)
    softmax = partial(torch.softmax, dim=1)
    with torch.no_grad():
        for img_sz, img_pths in testData.img_cluster.items():
            
            if img_sz == (512,512,3):
                batch_size = BATCH_SIZE
            elif img_sz == (1048,1048,3):
                batch_size = 4
            elif img_sz == (2048,2048,3):
                batch_size = 2
                
            pipe = simple_pipeline(img_pths, batch_size=batch_size, num_threads=2, device_id=0)
            pipe.build()

            test_dataloader = DALIGenericIterator(pipe, output_map=['image', 'idx'], size=pipe.epoch_size("Reader"), last_batch_policy=LastBatchPolicy.PARTIAL)

            for data in tqdm(test_dataloader, desc=f"{img_sz}", ascii=True, ncols=60):
                imgs = data[0]['image']
                idxs = data[0]['idx']
                imgs = imgs.permute(0,3,1,2)

                with autocast():
                    m_logits = softmax(model(imgs)) 
                
                m_logits = torch.argmax(m_logits, dim=1).cpu().numpy()
                for i in range(len(m_logits)):
                    img_ind = idxs[i]
                    cls_map = m_logits[i].astype(np.uint8) * 255
                    img_pth = testData.img_cluster[img_sz][img_ind]
                    img_nm = os.path.basename(img_pth).split('.')[0]
                    
                    threadpool.submit(save_img,SAVE_DIR,img_nm,cls_map)

if __name__ == '__main__':

    main()



