import time
from dataset import ColorDataset
from utils import utils
from utils import visualizer

import options
import torch
import torchvision
import torchvision.transforms as transforms

import argparse

from model import create_model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = options.initialize(parser)

    parser.add_argument('--dataset_root', default='datasets/train')
    parser.add_argument('--img_root', default='trainlist.txt')
    parser.add_argument('--Batch_size', default=12, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--gpu', default='0,1')
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--iter', default=5)
    parser.add_argument('--iter_decay', default=1)
    parser.add_argument('--isTrain', default=True)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--start_epoch', default='latest')
    parser.add_argument('--loadSize', default=256)
    parser.add_argument('--fineSize', default=176)

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


    dataset_root = args.dataset_root
    img_root = args.img_root

    # create dataset
    dataset = ColorDataset(args)
    print('#training images=%d' % len(dataset))

    Batch_size = args.Batch_size
    shuffle=args.shuffle
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=Batch_size, shuffle=shuffle)

    #create model
    model = create_model(args)
    model.setup(args)
    model.print_networks(True)


    visualizer = visualizer.Visualizer(args)
    total_steps = 0

    # train
    iters = args.iter
    iters_decay = args.iter_decay
    for epoch in range(args.epoch_count, iters):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, img in enumerate(dataLoader):
            image = img['img'].cuda()
            data = utils.get_colorization_data(image, args, num_points=5)
            if(data is None):
                print('no data')
                continue
            iter_start_time = time.time()
            if total_steps % args.print_freq == 0:
                # time to load data
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += args.Batch_size
            epoch_iter += args.Batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % args.print_freq == 0:
                save_result = total_steps % args.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % args.print_freq == 0:
                losses = model.get_current_losses()
                # time to do forward&backward
                # t = (time.time() - iter_start_time) / args.batch_size
                t = time.time() - iter_start_time
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                print('current learning rate: %.10f' % args.lr)
                if args.display_id > 0:
                    # embed()
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), args, losses)

            if total_steps % args.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % args.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, iters + iters_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
