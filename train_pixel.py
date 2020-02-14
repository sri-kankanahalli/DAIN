import sys
import os
import random

import threading
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *
from scipy.misc import imread, imsave, imresize

import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
import numpy as np
from my_args import args, unique_id

class Discriminator(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_chans, out_chans, downsample = False, dilation = 1):
            super(Discriminator.ResidualBlock, self).__init__()

            kernel_size = 3
            padding_amt = dilation * (kernel_size - 1) // 2

            self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size = kernel_size,
                                   padding = padding_amt, dilation = dilation,
                                   stride = 2 if downsample else 1)
            self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size = kernel_size,
                                   padding = padding_amt, dilation = dilation)

            # all convs share a common ReLU unit
            self.relu = nn.LeakyReLU(0.3)

            self.shortcut_conv = None
            if (downsample or in_chans != out_chans):
                self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size = kernel_size,
                                               padding = padding_amt, dilation = dilation,
                                               stride = 2 if downsample else 1)

        def forward(self, x):
            residual = x
            if (self.shortcut_conv is not None):
                residual = self.shortcut_conv(residual)
                residual = self.relu(residual)

            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.relu(out)

            out += residual

            return out

    def __init__(self):
        super(Discriminator, self).__init__()

        # number of starting feature maps
        ndf = 32

        self.main = nn.Sequential(
            Discriminator.ResidualBlock(3, ndf, dilation = 1),
            nn.MaxPool2d(2, padding = 1),
            Discriminator.ResidualBlock(ndf, ndf, dilation = 1),
            #Discriminator.ResidualBlock(ndf, ndf, dilation = 1),
            nn.MaxPool2d(2, padding = 1),
            Discriminator.ResidualBlock(ndf, ndf, dilation = 1),
            #Discriminator.ResidualBlock(ndf, ndf, dilation = 1),
            nn.MaxPool2d(2, padding = 1),
            Discriminator.ResidualBlock(ndf, ndf, dilation = 1),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(ndf, 1),
            #nn.Sigmoid()
        )

        # weight initialization function
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal(m.weight, 0.0, 0.02)
                m.bias.data.fill_(0.0)
            elif type(m) == nn.Conv2d:
                nn.init.normal(m.weight, 0.0, 0.02)
                m.bias.data.fill_(0.0)

        #self.apply(init_weights)


    def forward(self, input):
        return self.main(input)

# tensor of type 1xCxHxW
def tensor_to_numpy_img(X):
    X_numpy = X.cpu().data.numpy()[0]
    X_numpy = X_numpy.transpose((1, 2, 0))

    X_numpy = np.clip(X_numpy, 0.0, 1.0)
    X_numpy = np.round(X_numpy * 255.0).astype(np.uint8)
    return X_numpy

def charbonnier_loss(y_est, y):
        y_diff = y_est - y
        pixel_loss = torch.mean(torch.sqrt(y_diff * y_diff + args.epsilon * args.epsilon))
        return pixel_loss

def train():
    SAVED_MODEL_PATH = "./model_weights/pretrained.pth"
    DATA_PATH = "./pixel_triplets/"
    BATCH_SIZE = 1

    torch.manual_seed(1337)
    random.seed(1337)

    # -------------------------------------
    #  load pre-trained model
    # -------------------------------------
    model = networks.DAIN(channel = 3,
                          filter_size = 4,
                          timestep = 0.5,
                          training = False,
                          pixel_model = True)
    model = model.cuda()

    print("Fine tuning on " + SAVED_MODEL_PATH)

    pretrained_dict = torch.load(SAVED_MODEL_PATH)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    pretrained_dict = []

    # -------------------------------------
    #  create discriminator
    # -------------------------------------
    discrim = Discriminator()
    discrim = discrim.cuda()

    # discriminator optimizer and loss
    optimizer_discrim = torch.optim.Adam(discrim.parameters(), lr = 0.0005)

    # -------------------------------------
    #  create dataset loaders
    # -------------------------------------
    train_set, test_set = datasets.pixel_triplets(DATA_PATH)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = BATCH_SIZE,
        sampler = balancedsampler.RandomBalancedSampler(train_set, int(len(train_set) / BATCH_SIZE )),
        num_workers = 8, pin_memory = True)

    val_loader = torch.utils.data.DataLoader(test_set, batch_size = BATCH_SIZE,
                                             num_workers = 8, pin_memory = True)

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))

    # -------------------------------------
    #  create optimizer / LR scheduler
    # -------------------------------------
    print("train the interpolation net")
    optimizer = torch.optim.Adamax([
                {'params': model.initScaleNets_filter.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.initScaleNets_filter1.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.initScaleNets_filter2.parameters(), 'lr': args.filter_lr_coe * args.lr},
                {'params': model.ctxNet.parameters(), 'lr': args.ctx_lr_coe * args.lr},
                {'params': model.flownets.parameters(), 'lr': args.flow_lr_coe * args.lr},
                {'params': model.depthNet.parameters(), 'lr': args.depth_lr_coe * args.lr},
                {'params': model.rectifyNet.parameters(), 'lr': args.rectify_lr}
            ],
                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=args.factor, patience=args.patience,verbose=True)

    # -------------------------------------
    #  print out some info before we start
    # -------------------------------------
    print("*********Start Training********")
    print("LR is: "+ str(float(optimizer.param_groups[0]['lr'])))
    print("EPOCH is: "+ str(int(len(train_set) / BATCH_SIZE )))
    print("Num of EPOCH is: "+ str(args.numEpoch))

    def count_network_parameters(model):
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([numpy.prod(p.size()) for p in parameters])

        return N

    print("Num. of model parameters is:", count_network_parameters(model))
    if hasattr(model,'flownets'):
        print("Num. of flow model parameters is:",
              count_network_parameters(model.flownets))
    if hasattr(model,'initScaleNets_occlusion'):
        print("Num. of initScaleNets_occlusion model parameters is:",
              count_network_parameters(model.initScaleNets_occlusion) +
              count_network_parameters(model.initScaleNets_occlusion1) +
              count_network_parameters(model.initScaleNets_occlusion2))
    if hasattr(model,'initScaleNets_filter'):
        print("Num. of initScaleNets_filter model parameters is:",
              count_network_parameters(model.initScaleNets_filter) +
              count_network_parameters(model.initScaleNets_filter1) +
              count_network_parameters(model.initScaleNets_filter2))
    if hasattr(model, 'ctxNet'):
        print("Num. of ctxNet model parameters is:",
              count_network_parameters(model.ctxNet))
    if hasattr(model, 'depthNet'):
        print("Num. of depthNet model parameters is:",
              count_network_parameters(model.depthNet))
    if hasattr(model,'rectifyNet'):
        print("Num. of rectifyNet model parameters is:",
              count_network_parameters(model.rectifyNet))
    print("Num. of discriminator model parameters is:",
          count_network_parameters(discrim))

    # -------------------------------------
    #  and heeere we go
    # -------------------------------------

    # discriminator pretrains for a certain # of epochs
    PRETRAINING_EPOCHS = 0

    training_losses = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    ikk = 0
    for kk in optimizer.param_groups:
        if kk['lr'] > 0:
            ikk = kk
            break

    d_real_label = Variable(torch.ones(1,), requires_grad = False).cuda() * 0.5
    d_fake_label = Variable(torch.ones(1,), requires_grad = False).cuda() * -0.5

    g_label_target = Variable(torch.stack([
        d_real_label,
        d_real_label,
        d_real_label,
        d_real_label,
    ], dim = 0), requires_grad = False).cuda()

    d_label_target = Variable(torch.stack((
        d_real_label,
        d_real_label,
        d_real_label,
        d_real_label,
        d_fake_label,
        d_fake_label,
        d_fake_label,
        d_fake_label,
    ), dim = 0), requires_grad = False).cuda()

    for t in range(args.numEpoch):
        print("The id of this in-training network is " + unique_id)

        if (t < PRETRAINING_EPOCHS):
            print("-- Discriminator pre-training epoch --")
        elif (t == PRETRAINING_EPOCHS):
            print("-- End discriminator pre-training --")

        # turn into training mode
        model = model.train()
        discrim = discrim.train()

        for i, (X0_half, X1_half, y_half) in enumerate(train_loader):
            loss_function = charbonnier_loss

            #if i >= 100:# 
            if i >= int(len(train_set) / BATCH_SIZE):
                break

            if (t < PRETRAINING_EPOCHS and i >= 100):
                break

            #before_mod = sum([torch.mean(p) for p in model.parameters()]).item()
            #before_dsc = sum([torch.mean(p) for p in discrim.parameters()]).item()

            X0_half = X0_half.cuda()
            X1_half = X1_half.cuda()
            y_half = y_half.cuda()

            X0 = Variable(X0_half, requires_grad = False)
            X1 = Variable(X1_half, requires_grad = False)
            y  = Variable(y_half,requires_grad = False)

            # placeholder variables
            discrim_total_loss = Variable(torch.zeros(1, 1)).cuda()
            model_pixel_loss = torch.zeros(1,)
            model_dsc_loss = torch.zeros(1,)
            total_loss = torch.zeros(1,)

            # --------------------------------------------
            #  train the interpolation network
            #      (using the cycle consistency method from Reda, et al.)
            # --------------------------------------------
            optimizer.zero_grad()

            def create_model_input(before, after):
                return torch.cat((
                    torch.stack((before, after), dim = 0),
                ), dim = 1)

            # predict the frame between X0 and y
            model_input = torch.stack((X0, X1), dim = 0)
            model_output = model(model_input)

            y_est = model_output[0:1]

            # concatenate real and fake so we can do everything in two forward passes
            discrim_batch = torch.cat((
                y.detach(),
                y_est.detach()
            ), dim = 0)

            if (t >= PRETRAINING_EPOCHS):
                # pixel losses
                model_pixel_loss = charbonnier_loss(y_est, y)

                # discriminator losses
                C_y, C_y_est = discrim(discrim_batch)

                # RaLSGAN loss. what is it minimizing? uhhh i don't fucking know man
                C_diff = ((C_y - C_y_est - 1.0) ** 2 + (C_y_est - C_y + 1.0) ** 2) / 2.0
                model_dsc_loss = C_diff

                total_loss = model_pixel_loss + 0.01 * model_dsc_loss

                total_loss.backward()
                optimizer.step()

            # --------------------------------------------
            #  train the discriminator
            # --------------------------------------------
            optimizer_discrim.zero_grad()

            C_y, C_y_est = discrim(discrim_batch)

            # discriminator's RaLSGAN loss is the reverse of the generator's
            C_diff = ((C_y_est - C_y - 1.0) ** 2 + (C_y - C_y_est + 1.0) ** 2) / 2.0
            discrim_loss = C_diff

            discrim_loss.backward()
            optimizer_discrim.step()

            # --------------------------------------------
            #  finally, output some stuff
            # --------------------------------------------
            training_losses.update(total_loss.item(), BATCH_SIZE)
            if i % max(1, int(int(len(train_set) / BATCH_SIZE) / 500.0)) == 0:

                print("Ep [" + str(t) +"/" + str(i) +
                                    "]\tl.r.: " + str(round(float(ikk['lr']), 7))+
                                    "\tPix: " + str([round(model_pixel_loss.item(),5)]) +
                                    #"\tFool: " + str(100 - round(np.sqrt(model_dsc_loss.item()) * 100, 5)) + "%" +
                                    "\tFool: " + str(round(model_dsc_loss.item(), 5)) +
                                    "\tTotal: " + str([round(x.item(), 5) for x in [total_loss]]) +
                                    #"\tDiscrim: " + str(100 - round(np.sqrt(discrim_loss.item()) * 100, 5)) + "%" +
                                    "\tDiscrim: " + str(round(discrim_loss.item(), 5)) +
                                    "\tAvg. Loss: " + str([round(training_losses.avg, 5)]))

        if (t < PRETRAINING_EPOCHS):
            continue

        torch.save(model.state_dict(), args.save_path + "/epoch" + str(t) +".pth")

        # print("\t\t**************Start Validation*****************")

        #Turn into evaluation mode

        val_total_losses = AverageMeter()
        val_total_pixel_loss = AverageMeter()
        val_total_PSNR_loss = AverageMeter()

        for i, (X0,X1,y) in enumerate(val_loader):
            if i >=  int(len(test_set)/ BATCH_SIZE):
                break

            with torch.no_grad():
                X0 = X0.cuda()
                X1 = X1.cuda()
                y = y.cuda()

                y_est = model(torch.stack((X0, X1),dim = 0))

                y_diff = y_est - y
                pixel_loss = torch.mean(torch.sqrt(y_diff * y_diff + args.epsilon * args.epsilon))

                val_total_loss = pixel_loss

                per_sample_pix_error = torch.mean(torch.mean(torch.mean(y_diff ** 2,
                                                                    dim=1),dim=1),dim=1)
                per_sample_pix_error = per_sample_pix_error.data # extract tensor
                psnr_loss = torch.mean(20 * torch.log(1.0/torch.sqrt(per_sample_pix_error)))/torch.log(torch.Tensor([10]))

                val_total_losses.update(val_total_loss.item(), BATCH_SIZE)
                val_total_pixel_loss.update(pixel_loss.item(), BATCH_SIZE)
                val_total_PSNR_loss.update(psnr_loss[0], BATCH_SIZE)
                print(".",end='',flush=True)

        print("\nEpoch " + str(int(t)) +
              "\tlearning rate: " + str(float(ikk['lr'])) +
              "\tAvg Training Loss: " + str(round(training_losses.avg,5)) +
              "\tValidate Loss: " + str([round(float(val_total_losses.avg), 5)]) +
              "\tValidate PSNR: " + str([round(float(val_total_PSNR_loss.avg), 5)]) +
              "\tPixel Loss: " + str([round(float(val_total_pixel_loss.avg), 5)])
              )

        auxiliary_data.append([t, float(ikk['lr']),
                                   training_losses.avg, val_total_losses.avg, val_total_pixel_loss.avg])

        numpy.savetxt(args.log, numpy.array(auxiliary_data), fmt='%.8f', delimiter=',')
        training_losses.reset()

        print("\t\tFinished an epoch, Check and Save the model weights")
            # we check the validation loss instead of training loss. OK~
        if saved_total_loss >= val_total_losses.avg:
            saved_total_loss = val_total_losses.avg
            torch.save(model.state_dict(), args.save_path + "/best"+".pth")
            print("\t\tBest Weights updated for decreased validation loss\n")

        else:
            print("\t\tWeights Not updated for undecreased validation loss\n")

        #schdule the learning rate
        #scheduler.step(val_total_losses.avg)


    print("*********Finish Training********")

if __name__ == '__main__':
    sys.setrecursionlimit(100000)# 0xC00000FD exception for the recursive detach of gradients.
    threading.stack_size(200000000)# 0xC00000FD exception for the recursive detach of gradients.
    thread = threading.Thread(target=train)
    thread.start()
    thread.join()

    exit(0)
