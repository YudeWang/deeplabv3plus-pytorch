Most recurrent update:
2018.9.28 - add python evaluation func in ./lib/datasets/VOCDataset.py
2018.9.21 - fix some bugs in ./lib/datasets

# DeepLabv3plus Semantic Segmentation in Pytorch

Here is a pytorch implementation of deeplabv3+. The project supports COCO object detection dataset pretrain(transform to segmentation manually) and PASCAL VOC 2012 train/val. SynchronizedBatchNorm(from [vacancy](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)) has been used in modified ResNet backbone. Now the project achieve 79.161% on VOC2012 validation set with resnet101 and pretrained on COCO.

## Configuration Environment

Ubuntu16.04 + python3.6 + pytorch0.4 + CUDA8.0

You should run the code on more than 2 GPU because only multi-GPU version is supported at present.

Anaconda virtual environment is suggested.

Please install [tensorboardX](https://github.com/lanpa/tensorboardX) for loss curves and segmentation visualization. 

## Dataset
1. Download VOC2012 dataset and VOCdevkit, keep the VOC2012 folder in VOCdevkit as $root/data/VOCdevkit/VOC2012/...., or you can modified path in `$root/lib/datasets/VOCDataset.py`

2. Download VOC augmented segmentation dataset from [DrSleep](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0). Put the annotations in $root/data/VOCdevkit/VOC2012/SegmentationClass/ and replace training set file `$root/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt`

3. Download COCO2017 object detection dataset [here](http://cocodataset.org/#download) for pretrain, keep the path as $root/data/MSCOCO/...., or change it in`$root/lib/datasets/COCODataset.py`

4. Install Matlab for VOC2012 evaluation(we use VOC official matlab evaluation code in VOCdevkit)

5. (optional) You can also download [ADE20K dataset](http://sceneparsing.csail.mit.edu/) as same as VOC and COCO, but ADE20K dataset interface has not been finished yet.

## Network

Download COCO pretrained network parameter [here](https://drive.google.com/open?id=1nuEAm9JPT3J7MEqLYfWG7TiVQba5W223) and VOC2012 augmented dataset finetuned parameter [here](https://drive.google.com/open?id=1QM7R845GOo0T10fbqdzrOPhQep2ttDoy). Place them in $root/model/...

We download imagenet pretrained models by URLs when load them.  

The backbone only support resnet at present. Though Xception has been finished, it has not been tested yet.

For ResNet backbone, the structure is modified according to [2] which repeats layer4 four times with atrous rate as 1,2,1

## Train & Test

Please check the configuration file `$root/experiment/project_name/config.py` first to meet your requires.

Please set visible gpu devices before training.

```
export CUDA_VISIBLE_DEVICES=0,1
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
- [x] model pretrained on COCO
- [x] flip test
- [ ] xception as backbone
- [ ] multiscale test

The project achieves 79.161% on VOC2012 validation set with resnet101 and pretrained on COCO.



## References

1. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. In ECCV, 2018.

2. Rethinking Atrous Convolution for Semantic Image Segmentation
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1706.05587, 2017.
