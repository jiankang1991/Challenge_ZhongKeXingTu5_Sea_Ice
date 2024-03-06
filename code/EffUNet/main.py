
import argparse
from datetime import datetime
import os
import yaml
import sys
sys.path.append('../')

from utils.config import parse
from train import Trainer
# from train_mlt import Trainer
# from train_v1 import Trainer

def defineyaml(args):
    #YAML
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    yamlcontents = f"""
sv_name: '{sv_name}'
model_name: unet
# model_path: ../../zkxt21_SAR/EffUNet/20210919_100730/checkpoints/20210919_100730_checkpoint.pth.tar
model_path: 
train: true
img_dir: {args.sarimg_dir}
msk_dir: {args.sarmsk_dir}
compare_model:
    # model_name: unet
    # model_name: Linknet
    # model_name: pspnet
    # model_name: fpn
    # model_name: deeplabv3
    # model_name: deeplabv3+
model_specs:
    # encoder_name: timm-res2net50_26w_4s
    # encoder_name: dpn68
    # encoder_name: efficientnet-b4
    # encoder_name: efficientnet-b5
    # encoder_name: se_resnext50_32x4d
    # encoder_name: timm-efficientnet-b4
    # encoder_name: timm-tf_efficientnet_lite4
    encoder_name: timm-tf_efficientnet_lite3
    in_channels: 3
    classes: 2
    # encoder_depth: 5
    encoder_depth: 4
    decoder_channels:
        # - 256 
        - 128
        - 64
        - 32
        - 16
    decoder_attention_type: scse
batch_size: 8
data_specs:
    width: 512
    height: 512
    dtype:
    rescale: false
    rescale_minima: auto
    rescale_maxima: auto
    label_type: mask
    is_categorical: false
    mask_channels: 1
    val_holdout_frac:
    data_workers: 4
training_data_csv: {args.trainsarcsv}
validation_data_csv: {args.validsarcsv}

training_augmentation:
    augmentations:
        HorizontalFlip:
            p: 0.5
        # VerticalFlip:
        #     p: 0.5
        RandomRotate90:
            p: 0.5
        Normalize:
            mean:
                - 0.5
            std:
                - 0.125
            max_pixel_value: 255.0
            p: 1.0
    p: 1.0
    shuffle: true
validation_augmentation:
    augmentations:
        Normalize:
            mean:
                - 0.5
            std:
                - 0.125
            max_pixel_value: 255.0
            p: 1.0
    p: 1.0
training:
    epochs: 150
    lr: 1e-3
    loss:
        diceloss:
            mode: multiclass
            from_logits: True
        crossentropyloss:
        boundaryloss:
            cls_num: 2
    loss_weights:
        diceloss: 1.0
        crossentropyloss: 1.0
        boundaryloss: 1.0
"""
    print('saving file name is ', sv_name)
    checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
    logs_dir = os.path.join('./', sv_name, 'logs')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    with open(os.path.join('./', sv_name, f'{sv_name}.yaml'), 'w') as f:
        f.write(yamlcontents)
    
    return sv_name, checkpoint_dir, logs_dir

def defineoptyaml(args):
    #YAML
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    yamlcontents = f"""
sv_name: '{sv_name}'
model_name: unet
model_path:
train: true
img_dir: {args.optimg_dir}
msk_dir: {args.optmsk_dir}
model_specs:
    encoder_name: efficientnet-b5
    in_channels: 3
    classes: 2
    encoder_depth: 5
    decoder_channels: 
        - 128
        - 128
        - 64
        - 32
        - 16
    decoder_attention_type: scse
batch_size: 16
data_specs:
    width: 256
    height: 256
    dtype:
    image_type: 32bit
    rescale: false
    rescale_minima: auto
    rescale_maxima: auto
    label_type: mask
    is_categorical: false
    mask_channels: 1
    val_holdout_frac:
    data_workers: 4
training_data_csv: {args.trainoptcsv}
validation_data_csv: {args.validoptcsv}
training_augmentation:
    augmentations:
        HorizontalFlip:
            p: 0.5
        RandomRotate90:
            p: 1.0
        RandomCrop:
            height: 512
            width: 512
            p: 1.0
        Normalize:
            mean:
            - 0.5
            std:
            - 0.125
            max_pixel_value: 255.0
            p: 1.0
    p: 1.0
    shuffle: true
validation_augmentation:
    augmentations:
        CenterCrop:
            height: 512
            width: 512
            p: 1.0
        Normalize:
            mean:
            - 0.5
            std:
            - 0.125
            max_pixel_value: 255.0
            p: 1.0
    p: 1.0
training:
    epochs: 150
    lr: 5e-3
    loss:
        diceloss:
            mode: multiclass
            from_logits: True
        focalloss:
            mode: multiclass
    loss_weights:
        diceloss: 1.0
        focalloss: 10.0
    """
    
    print('saving file name is ', sv_name)
    checkpoint_dir = os.path.join('./', sv_name, 'checkpoints')
    logs_dir = os.path.join('./', sv_name, 'logs')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    with open(os.path.join('./', sv_name, f'{sv_name}.yaml'), 'w') as f:
        f.write(yamlcontents)

    return sv_name, checkpoint_dir, logs_dir

def main(args):

    if not args.opt_pretrain and not args.only_opt:
        sv_name, checkpoint_dir, logs_dir = defineyaml(args)
        config = parse(os.path.join('./', sv_name, f'{sv_name}.yaml'))
        config['checkpoint_dir'] = checkpoint_dir
        config['logs_dir'] = logs_dir
        trainer = Trainer(config)
        trainer.run()
    elif args.opt_pretrain and not args.only_opt:
        opt_sv_name, opt_checkpoint_dir, opt_logs_dir = defineoptyaml(args)
        opt_config = parse(os.path.join('./', opt_sv_name, f'{opt_sv_name}.yaml'))
        opt_config['checkpoint_dir'] = opt_checkpoint_dir
        opt_config['logs_dir'] = opt_logs_dir
        trainer = Trainer(opt_config)
        trainer.run()

        sar_sv_name, sar_checkpoint_dir, sar_logs_dir = defineyaml(args)
        sar_config = parse(os.path.join('./', sar_sv_name, f'{sar_sv_name}.yaml'))
        sar_config['checkpoint_dir'] = sar_checkpoint_dir
        sar_config['logs_dir'] = sar_logs_dir
        sar_config['model_path'] = os.path.join(opt_checkpoint_dir, opt_sv_name + '_checkpoint.pth.tar')

        with open(os.path.join('./', sar_sv_name, f'{sar_sv_name}.yaml'), 'w') as f:
            yaml.dump(sar_config, f)
        trainer = Trainer(sar_config)
        trainer.run()
    elif not args.opt_pretrain and args.only_opt:
        opt_sv_name, opt_checkpoint_dir, opt_logs_dir = defineoptyaml(args)
        opt_config = parse(os.path.join('./', opt_sv_name, f'{opt_sv_name}.yaml'))
        opt_config['checkpoint_dir'] = opt_checkpoint_dir
        opt_config['logs_dir'] = opt_logs_dir
        trainer = Trainer(opt_config)
        trainer.run()
        
    return None

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SpaceNet 6 Algorithm')
    parser.add_argument('--sarimg-dir', default=None,
                        help='Where the sar training img')
    parser.add_argument('--sarmsk-dir', default=None,
                        help='Where the sar training msk')
    parser.add_argument('--optimg-dir', default=None,
                        help='Where the opt training img')
    parser.add_argument('--optmsk-dir', default=None,
                        help='Where the opt training msk')                    
    parser.add_argument('--trainsarcsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/train.csv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--validsarcsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/valid.csv',
                        help='Where to save reference CSV of validation data')
    parser.add_argument('--trainoptcsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/train.csv',
                        help='Where to save reference CSV of training data')
    parser.add_argument('--validoptcsv', default='/home/zkgy/Data/SpaceNet6/proc_train_test/valid.csv',
                        help='Where to save reference CSV of validation data')
    parser.add_argument('--opt-pretrain', action='store_true',
                        help='Pretrain models on optical images')
    parser.add_argument('--only-opt', action='store_true',
                        help='train models on optical images')                    
    args = parser.parse_args()

    main(args)