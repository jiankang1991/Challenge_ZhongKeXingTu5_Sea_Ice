## 环境安装
```
#创建pytorch环境
conda create -n pytorch python=3.8  
conda activate pytorch

#安装pytorch
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=11.0 -c pytorch

pip install -r requirements.txt
```



## 训练 & 测试

### 数据准备
- 采用3折交叉训练的方法，将训练集和测试集分为3组，保存为csv文件
- 训练集包含1825张图像，测试集包括203张图像

csv文件保存在data文件夹中，文件结构如下所示: 
```
|-- data
    |-- train_0.csv
    |-- train_1.csv
    |-- train_2.csv
    |-- val_0.csv
    |-- val_1.csv
    |-- val_2.csv
```
### 数据处理

- 首先将大于512$\times$512的图像统一裁剪为512$\times$512的大小，以确保一致性。
- 网络的训练阶段，采用水平翻转以及随机90°旋转作为数据增强，并用平均值和标准偏差值分别为127.5和31.875对图像进行了归一化。

### 训练

实验中网路采用time-EfficientNetlite3作为backbone的Unet架构，通过执行以下命令运行train.py文件：

```
cd EffUNet
. run.sh
```
run.sh文件中已经包含了train.py文件需要输入设置的参数：
```
--sarimg-dir  #训练图像路径
--sarmsk-dir  #标签路径
--trainsarcsv #训练图像csv文件路径
--validsarcsv #测试图像csv文件路径
```

EffUNet/model.py文件中包含了实验中的模型以及对比的模型



### 测试

通过执行以下命令运行test_论文.py文件：

```
cd test 
python test_论文.py
```

test_论文.py文件中需要更改的参数如下：

```
--BATCH_SIZE  #批次大小
--INPUT_DIR   #测试图像路径
--SAVE_DIR    #预测图像保存路径
--config      #参数配置文件路径
```



### 模型性能评估

用于评估性能指标的文件保存在test文件夹中，文件结构如下：

```
|-- test
	|-- test_论文.py
    |-- test.py
    |-- img_preds_vis.ipynb
    |-- test_metrics.ipynb
```

- test_论文.py用于论文中的模型预测
- test.py用于比赛中的模型预测
- img_preds_vis.ipynb用于可视化模型预测图
- test_metrics.ipynb 用于评估模型性能

论文模型参数保存在**/boot/data1/Li_data/data/毕业论文_模型参数/zkxt**

模型预测图保存在各自文件夹的**test_predict**下



## 自定义函数

自定义函数保存在utils文件夹中，文件如下：

```
|-- utils
    |-- config.py
    |-- core.py
    |-- dataGen.py
    |-- io.py
    |-- losses.py
    |-- metrics.py
    |-- modules.py
    |-- transform_n.py
```

- utils/config.py用于读取配置文件
- utils/core.py用于检查文件是否加载
- utils/dataGen.py用于数据处理
- utils/io.py用于读取图片
- utils/losses.py中包含**损失函数**
- utils/metrics.py中包含**评估指标**
- utils/modules.py中包含实验中的各种**模块**
- utils/transform_n.py中包含**数据增强**操作