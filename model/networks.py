import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.iter) / float(opt.iter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='xavier', gpu_ids=[], use_tanh=True, classification=True):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)
    netG = SIGGRAPHGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=use_tanh, classification=classification)
    return init_net(netG, init_type, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='xavier', gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, gpu_ids)


##############################################################################
# Classes
##############################################################################


class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta=delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0-in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl*mask/self.delta + (mann-.5*self.delta)*(1-mask)
        return torch.sum(loss,dim=1,keepdim=True)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum(torch.abs(in0-in1),dim=1,keepdim=True)

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum((in0-in1)**2,dim=1,keepdim=True)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class SIGGRAPHGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=True):
        super(SIGGRAPHGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True
        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation
        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above        

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # classification output
        model_class=[nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B, mask_B):
        conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if(self.classification):
            out_class = self.model_class(conv8_3)
            #conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            #conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

        return (out_class, out_reg)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
