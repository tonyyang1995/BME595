import argparse
import os
import torch
import model

# the original pix2pix has a lot of options
# I don't change them in my project
# So to make it suitable to implement, I just put it here and use the default values
def initialize(parser):
    parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--name', type=str, default='color_face', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
    parser.add_argument('--model', type=str, default='pix2pix',
                            help='chooses which model to use. cycle_gan, pix2pix, test')
    parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization')
    parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--no_dropout', action='store_true', default=True, help='no dropout for the generator')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
    parser.add_argument('--ab_norm', type=float, default=110., help='colorization normalization factor')
    parser.add_argument('--ab_max', type=float, default=110., help='maximimum ab value')
    parser.add_argument('--ab_quant', type=float, default=10., help='quantization factor')
    parser.add_argument('--l_norm', type=float, default=100., help='colorization normalization factor')
    parser.add_argument('--l_cent', type=float, default=50., help='colorization centering factor')
    parser.add_argument('--mask_cent', type=float, default=.5, help='mask centering factor')
    parser.add_argument('--sample_p', type=float, default=0.9, help='sampling geometric distribution, 1.0 means no hints')
    parser.add_argument('--sample_Ps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, ], help='patch sizes')

    parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
    parser.add_argument('--classification', action='store_true', default=True, help='backprop trunk using classification, otherwise use regression')
    parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
    parser.add_argument('--how_many', type=int, default=200, help='how many test images to run')
    parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')

    parser.add_argument('--half', action='store_true', default=False, help='half precision model')
    
    parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--update_html_freq', type=int, default=1, help='frequency of saving training results to html')
    parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--no_lsgan', action='store_true', default=True, help='do *not* use least square GAN, if false, use vanilla GAN')
    parser.add_argument('--lambda_GAN', type=float, default=0., help='weight for GAN loss')
    parser.add_argument('--lambda_A', type=float, default=1., help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=1., help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.'
                        'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
    parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
    parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=500, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--avg_loss_alpha', type=float, default=.986, help='exponential averaging weight for displaying loss')

    return parser
