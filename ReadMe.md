## Requirements
[pytorch 0.4](https://pytorch.org/)
Install [visdom](https://github.com/facebookresearch/visdom)

## Dataset prepared:
1.0 [ImageNet Detection](http://www.image-net.org/)
2.0 [ImageNet Video Detection](http://www.image-net.org/)
3.0 Ami-face Dataset has been uploaded in my [google drive](https://drive.google.com/file/d/1qVpZqr_kmgQUdI25Gy0jYSBQ8f4i-PED/view?usp=sharing)

Put the datasets into the file "datasets"

## Prepare the train Lists and val Lists
use the script "create\_train\_list.py"

## Pre-train model
the color-face pretrain model has been uploaded in my [google drive](https://drive.google.com/file/d/13KlnL5bBbrEZoSneln8g_SoTvTrYau0R/view?usp=sharing)

## demo
```
python3 test.py
```
All the hyper parameters has been embed.

## train the model
1.0 download the pre-train model and datasets
2.0 run the sample command:
```
python3 train.py --dataset_root /path/to/ur/dataset --img_root /path/to/ur/imageLists --name /ur own name --Batch_size /ur own batchsize
```
for more hyper parameters, check the options.py

To view the training results, run 
```
python -m visdom.serve
``` 
and goto the remote [server](http://localhost:8097)

## Acknowledge
This code is based on the [pytorch-cycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [colorization-pytorch](https://github.com/richzhang/colorization-pytorch), [self-attention-gan](https://github.com/heykeetae/Self-Attention-GAN).

Thanks for my classmate BingBing Yang for sharing the animeface Dataset.
