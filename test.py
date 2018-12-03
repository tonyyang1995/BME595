import time
import os
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import argparse
import options

from dataset import ColorDataset
from utils import utils
from utils import visualizer
from utils import html

from model import create_model

if __name__ == '__main__':
    #sample_ps = [1., .125, .03125]
    sample_ps = [1.]
    to_visualize = ['gray', 'hint', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg',]
    #to_visualize = ['fake_reg']
    S = len(sample_ps)

    parser = argparse.ArgumentParser()
    parser = options.initialize(parser)
    parser.add_argument('--dataset_root', default='datasets/animeface_val')
    parser.add_argument('--img_root', default='Lists/animefaceVal.txt')
    parser.add_argument('--Batch_size', default=1)
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--iter', default=1)
    parser.add_argument('--iter_decay', default=500)
    parser.add_argument('--isTrain', default=True)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--start_epoch', default='latest')
    parser.add_argument('--loadSize', default=256)
    parser.add_argument('--findSize', default=176)
    # convert GPU from str to list
    args = parser.parse_args()

    if args.suffix:
        suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
        args.name = args.name + suffix
    # set gpu ids
    str_ids = args.gpu.split(',')
    args.gpu = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu.append(id)
    if len(args.gpu) > 0:
        torch.cuda.set_device(args.gpu[0])
    args.A = 2 * args.ab_max / args.ab_quant + 1
    args.B = args.A

    # set vars for test
    args.load_model=True
    args.nThreads = 1
    args.Batch_size = 1
    args.display_id= -1
    args.serial_batches = True
    args.aspect_ratio = 1.

    #dataset = ColorDataset(176, 176, args.dataset_root, args.img_root)
    dataset = datasets.ImageFolder(args.dataset_root,transforms.Compose([
                                    transforms.Resize((args.loadSize, args.loadSize)),
                                    transforms.ToTensor()]))
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.Batch_size, shuffle=False)
    model = create_model(args)
    model.setup(args)
    model.eval()

    # create website
    web_dir = os.path.join(args.results_dir, args.name, '%s_%s' % (args.phase, args.start_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (args.name, args.phase, args.start_epoch))

    # statistics
    #psnrs = np.zeros((args.how_many, S))
    #entrs = np.zeros((args.how_many, S))

    for i, img in enumerate(dataLoader):
        #image = img['img'].cuda()
        img[0] = img[0].cuda()
        # the data is already grayscale, there is no need split Lab
        img[0] = utils.crop_mult(img[0], mult=8)
        '''
        model.set_input(data)
        model.test(True)
        visuals = utils.get_subset_dict(model.get_current_visuals(), to_visualize)
        fake_reg = visuals['fake_reg']
        utils.save_image(fake_reg, os.path.join(args.results.dir, str(i))+'.png')

        '''
        # with no points
        for (pp, sample_p) in enumerate(sample_ps):
            img_path = [str.replace('%08d_%.3f' % (i, sample_p), '.', 'p')]
            data = utils.get_colorization_data(img[0], args, ab_thresh=0., p=sample_p)

            model.set_input(data)
            model.test(True)  # True means that losses will be computed
            visuals = utils.get_subset_dict(model.get_current_visuals(), to_visualize)

            #psnrs[i, pp] = utils.calculate_psnr_np(utils.tensor2im(visuals['real']), utils.tensor2im(visuals['fake_reg']))

            visualizer.save_images(webpage, visuals, img_path, aspect_ratio=args.aspect_ratio, width=args.display_winsize)
        if i % 10 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        #if i == args.how_many - 1:
        #    break
    webpage.save()

    # Compute and print some summary statistics
    #psnrs_mean = np.mean(psnrs, axis=0)
    #psnrs_std = np.std(psnrs, axis=0) / np.sqrt(args.how_many)

    #entrs_mean = np.mean(entrs, axis=0)
    #entrs_std = np.std(entrs, axis=0) / np.sqrt(args.how_many)

    #for (pp, sample_p) in enumerate(sample_ps):
    #    print('p=%.3f: %.2f+/-%.2f' % (sample_p, psnrs_mean[pp], psnrs_std[pp]))
