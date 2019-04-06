Most recurrent update:

2019.01.21 - Updata the code for paper performance achieved! Now deeplabv3+res101 achieve 79.155% and deeplabv3+xception achieve 79.945% on PASCAL VOC 2012 val set. The main bug is the missing of `patch_replication_callback()` function of Synchronized Batch Normalization.

2018.11.26 - Update including support Xception network, multi-scale test, network output stride modification, pure train set finetuning, and more dataset interface (PASCAL Context, Cityscapes, ADE20K)  

2018.09.28 - Add python evaluation func in `./lib/datasets/VOCDataset.py`

2018.09.21 - Fix some bugs in `./lib/datasets`

# DeepLabv3plus Semantic Segmentation in Pytorch

Here is a pytorch implementation of deeplabv3+. The project support variants of dataset including MS COCO object detection dataset, PASCAL VOC, PASCAL Context, Cityscapes, ADE20K. 


## Configuration Environment

Ubuntu16.04 + python3.6 + pytorch0.4 + CUDA8.0

You should run the code on more than 2 GPU because only multi-GPU version is supported at present.

Anaconda virtual environment is suggested.

Please install [tensorboardX](https://github.com/lanpa/tensorboardX) for loss curves and segmentation visualization. 

SynchronizedBatchNorm (from [vacancy](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)) has been used in our model, which will not be installed manually.

## Dataset
1. Download VOC2012 dataset and VOCdevkit, keep the VOC2012 folder in VOCdevkit as `$root/data/VOCdevkit/VOC2012/....`, or you can modified path in `$root/lib/datasets/VOCDataset.py`

2. Download VOC augmented segmentation dataset from [DrSleep](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). Put the annotations in `$root/data/VOCdevkit/VOC2012/SegmentationClass/` and name list file at `$root/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt`

3. Download COCO2017 object detection dataset [here](http://cocodataset.org/#download) for pretrain, keep the path as `$root/data/MSCOCO/....`, or change it in `$root/lib/datasets/COCODataset.py`

4. (optional) Install Matlab for VOC2012 evaluation (VOC official evaluation code in VOCdevkit is written by Matlab, and we have reimplement a python version)

5. (optional) You can also download [ADE20K dataset](http://sceneparsing.csail.mit.edu/), [Cityscapes](https://www.cityscapes-dataset.com), [PASCAL Context](https://cs.stanford.edu/~roozbeh/pascal-context/) as same as VOC and COCO, check the dataset path setting in `$root/lib/datasets/xxxDataset.py`.

## Network

Now the project support modified ResNet and Xception network as backbone. 

For ResNet backbone, the structure change the dilation of layer4 as [2,2,2].

As for Xception, we keep the same structure with offical tensorflow [code](https://github.com/tensorflow/models/tree/master/research/deeplab) and transform the pretrained parameter file from `.ckpt` to `.pth` manually!

Now ResNet101 and Xception achieve paper performance without multi-scale and flip test and finetuning on VOC2012aug dataset. The models with different setting will be released once paper performance is achieved. You can also finetuning the model by yourself where COCO dataset inferface has already be released. Discussion about finetuning tricks is welcomed by email(yude.wang@outlook.com) or issues!!!

Here are some pretrained model (46epoch, about 30k iterations) for download, the performance is evaluated on PASCAL VOC 2012 val set. (deeplabv3+ models are trained on multi GPUs, it may cause error when loading by cpu version):

| backbone | output stride | multi-scale & flip test | paper performance(mIoU) | our performance(mIoU) |
|----------|---------------|-------------------------|-------------------------|-----------------------|
| [xception_pytorch_imagnet](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi) | | | |
| [deeplabv3+res101](https://drive.google.com/open?id=1jSfvNDa60Kq5_KhoUuTKEQX-QbW4RCzn) | 16 | False | 78.85% | 79.155% |
|                   | 16 | True | 80.22% | 79.916% |
| [deeplabv3+xception](https://drive.google.com/open?id=11lgslZ4ayeYZTUQ99Ccu5hpgAWzfLPqj) | 16 | False | 79.93% | 79.945% |
|                   | 16 | True | 81.44% | 81.087% |


Model checkpoint with higher performance will be updata once achieve. 

## Train & Test

Please check the configuration file `$root/experiment/project_name/config.py` first to meet your requirements.

Please check the pretrained checkpoint path in `$root/lib/net/xception.py` and `$root/lib/net/resnet_atrous.py`

Please set visible gpu devices before training.

```
export CUDA_VISIBLE_DEVICES=0,1,2,3
``` 

Then you can train the network as you wish.

```
cd $root/experiment/project_name
python train.py
```

You can also test the performance of trained network by:

```
cd $root/experiment/project_name
python test.py
```

## Implementation Detail
- [x] deeper ResNet101 backbone
- [x] atrous convolution
- [x] multi-gpu support
- [x] synchronized batch normalization
- [x] decoder
- [ ] model pretrained on COCO
- [x] flip test
- [x] xception as backbone
- [x] multiscale test
- [x] achieve the performance mentioned in paper.

Discussion about learning tricks is welcomed (yude.wang@outlook.com).

## References

1. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. In ECCV, 2018.

2. Rethinking Atrous Convolution for Semantic Image Segmentation
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1706.05587, 2017.
