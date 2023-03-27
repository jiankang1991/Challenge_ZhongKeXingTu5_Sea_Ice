export CUDA_VISIBLE_DEVICES=5

python main.py \
--sarimg-dir /data/trainData/image \
--sarmsk-dir /data/trainData/gt \
--trainsarcsv ./data/train.csv \
--validsarcsv ./data/val.csv \


