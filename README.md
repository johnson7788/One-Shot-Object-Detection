# One-Shot Object Detection with Co-Attention and Co-Excitation

## Introduction

![Image](images/method.png)

[**One-Shot Object Detection with Co-Attention and Co-Excitation**](https://arxiv.org/abs/1911.12529)  
Ting-I Hsieh, Yi-Chen Lo, Hwann-Tzong Chen, Tyng-Luh Liu  
Neural Information Processing Systems (NeurIPS), 2019  
[slide](https://drive.google.com/drive/folders/1sTjR75hgDDML2owb9erRdnKVLRvuluo4), [poster](https://drive.google.com/drive/folders/1sTjR75hgDDML2owb9erRdnKVLRvuluo4)

This project is a pure pytorch implementation of *One-Shot Object Detection*. A majority of the code is modified from [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).  



### What we are doing and going to do

- [x] Support tensorboardX.
- [x] Upload the ImageNet pre-trained model.
- [x] Provide Reference image.
- [x] Provide checkpoint model.
- [ ] Train PASCAL_VOC datasets

## Preparation

First of all, clone the code

```bash
git clone https://github.com/timy90022/One-Shot-Object-Detection.git
```

### 1. Prerequisites

* Ubuntu 16.04
* Python or 3.6
* Pytorch 1.0

### 2. 准备数据集

* **COCO**: Please also follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the data.
See the scripts provided in this repository.
下载文件:
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar

解压：
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar

VOCdevkit2007/VOCcode/
VOCdevkit2007/VOC2007/

### 3. 预训练模型
我们在实验中使用ResNet50作为预训练的模型。这个预训练模型是通过排除所有与COCO相关的ImageNet类来训练的，
这是通过将ImageNet类的WordNet同义词与COCO类匹配来实现的。因此，我们只保留了其余725个类中的933,052张图片，
而原始数据集包含1000个类中的1,284,168张图片。预训练的模型在
ResNet50: [Google Drive](https://drive.google.com/file/d/1SL9DDezW-neieqxWyNlheNefwgLanEoV/view?usp=sharing)

Download and unzip them into the ../data/

### 4. Reference images
参考图像是通过裁剪出相对于Mask R-CNN的预测bounding boxes的斑块来检索的，而bounding boxes需要满足以下条件。

* The IOU threshold    > 0.5
* The score confidence > 0.7

参考图片可在以下网址获取:
* Reference file: [Google Drive](https://drive.google.com/file/d/1O1AQtjozgpdtuETGE6X4UItpqcVPUiXH/view?usp=sharing)

下载并解压到./data/.

### 5. Compilation

This step can be referred to [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch).
安装依赖

```bash
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:


```bash
cd lib
python setup.py build develop
```

报错：
/BHRL/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 386, in _check_cuda_version
    raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
RuntimeError: 
The detected CUDA version (10.1) mismatches the version that was used to compile
PyTorch (11.6). Please make sure to use the same CUDA versions.

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align, and ROI_Crop. The default version is compiled with Python 2.7. 

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**

## 训练
在训练之前，设置正确的目录来保存和加载训练后的模型。改变trainval_net.py和test_net.py的参数 "save_dir "和 "load_dir"，以适应你的环境。
在coco数据集中，我们把它分成4组。它将训练和测试不同的类别。只需调整 "*--g*"（1~4）。如果你想训练其他设置，你应该把 "*--g 0*"分开来。
如果你想训练数据集的部分内容，可以尝试修改"*--seen*". 

* 1 --> Training, session see train_categories(config file) class
* 2 --> Testing, session see test_categories(config file) class
* 3 --> session see train_categories + test_categories class

要在COCO上用ResNet50训练一个模型，只需运行

```bash
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                   --dataset coco --net res50 \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --cuda --g $SPLIT --seen $SEEN
```
以上，BATCH_SIZE和WORKER_NUMBER可以根据你的GPU内存大小自适应设置。
**在具有32G内存的NVIDIA V100 GPU上，它可以达到批次大小16**。
如果你有多个（比如说8个）V100 GPU，那么就把它们都用上吧! 尝试

```bash
python trainval_net.py --dataset coco --net res50 \
                       --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                       --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                       --cuda --g $SPLIT --seen $SEEN --mGPUs

```

## Test
如果你想评估ResNet50模型在COCO测试集上的检测性能，你可以自己训练或从[Google Drive](https://drive.google.com/file/d/1FV7TpTSgF0pwGxshqUSK-AvhXHSAObo4/view?usp=sharing)下载模型。
并将其解压到``./models/res50/``中。

Simply run

```bash
python test_net.py --dataset coco --net res50 \
                   --s $SESSION --checkepoch $EPOCH --p $CHECKPOINT \
                   --cuda --g $SPLIT
```

Specify the model session, checkepoch and checkpoint, e.g., SESSION=1, EPOCH=10, CHECKPOINT=1663.

If you want to test our model checkpoint, simple run  

For coco first group:

```bash
python test_net.py --s 1  --g 1 --a 4 --cuda
```

For coco second group:

```bash
python test_net.py --s 2  --g 2 --a 4 --cuda
```

## Acknowledgments

Code is based on [jwyang/faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch) and [AlexHex7/Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch). 

## Citation

```
@incollection{NIPS2019_8540,
  title 	= {One-Shot Object Detection with Co-Attention and Co-Excitation},
  author 	= {Hsieh, Ting-I and Lo, Yi-Chen and Chen, Hwann-Tzong and Liu, Tyng-Luh},
  booktitle 	= {Advances in Neural Information Processing Systems 32},
  year		= {2019},
  publisher 	= {Curran Associates, Inc.}
}

```
