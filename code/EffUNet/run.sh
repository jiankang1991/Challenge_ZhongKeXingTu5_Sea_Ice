export CUDA_VISIBLE_DEVICES=5

# [unet, Linknet, pspnet, fpn, deeplabv3, deeplabv3+]
python main.py \
--sarimg-dir /boot/data1/kang_data/zkxt21_seaice/trainData/image \
--sarmsk-dir /boot/data1/kang_data/zkxt21_seaice/trainData/gt \
--trainsarcsv ../data/train_2.csv \
--validsarcsv ../data/val_2.csv \


# python main.py \
# --sarimg-dir /boot/data1/Li_data/data/zkxt/train_image \
# --sarmsk-dir /boot/data1/Li_data/data/zkxt/train_gt \
# --compare_model 'Linknet' \
# --trainsarcsv /boot/data1/Li_data/data/zkxt/train2.csv \
# --validsarcsv /boot/data1/Li_data/data/zkxt/val2.csv \
# python main.py \
# --sarimg-dir /data/zkxt21_seaice/trainData/image \
# --sarmsk-dir /data/zkxt21_seaice/trainData/gt \
# --trainsarcsv ../data/train_1.csv \
# --validsarcsv ../data/val_1.csv \

# python main.py \
# --sarimg-dir /data/zkxt21_seaice/trainData/image \
# --sarmsk-dir /data/zkxt21_seaice/trainData/gt \
# --trainsarcsv ../data/train_2.csv \
# --validsarcsv ../data/val_2.csv \

