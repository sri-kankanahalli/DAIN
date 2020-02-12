import sys
import os
import random

import threading
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *

import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
from my_args import args, unique_id

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # number of starting feature maps
        ndf = 48

        self.main = nn.Sequential(
            # input size: 3 x 192 x 128
            nn.Conv2d(3, ndf, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.3, inplace=True),
            # after above: (ndf) x 96 x 64
            nn.Conv2d(ndf, ndf * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.3, inplace=True),
            # after above: (ndf*2) x 48 x 32
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.3, inplace=True),
            # after above: (ndf*4) x 24 x 16
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.3, inplace=True),
            # after above: (ndf*8) x 12 x 8
            nn.Conv2d(ndf * 8, ndf * 16, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.3, inplace=True),
            # after above: (ndf*16) x 6 x 4
            nn.Conv2d(ndf * 16, 1, kernel_size = 4, stride = 2, padding = 1),
            nn.LeakyReLU(0.3, inplace=True),
            # after above: 1 x 3 x 2
            nn.Flatten(),
            # after above: 6
            nn.Linear(6, 1),
            # after above: 1
            nn.Sigmoid()
        )

        # weight initialization function
        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.normal(m.weight, 0.0, 0.05)
                m.bias.data.fill_(0.01)
            elif type(m) == nn.Conv2d:
                nn.init.normal(m.weight, 0.0, 0.05)

        self.apply(init_weights)


    def forward(self, input):
        return self.main(input)


def train():
    SAVED_MODEL_PATH = "./model_weights/pretrained.pth"
    DATA_PATH = "./pixel_triplets/"
    BATCH_SIZE = 1

    torch.manual_seed(2020)

    # -------------------------------------
    #  load pre-trained model
    # -------------------------------------
    model = networks.DAIN(channel=args.channels,
                          filter_size = args.filter_size ,
                          timestep=args.time_step,
                          training=True,
                          pixel_model=True)
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
    pretrained_dict = None

    # -------------------------------------
    #  create discriminator
    # -------------------------------------
    discrim = Discriminator()
    discrim = discrim.cuda()

    # discriminator optimizer and loss
    optimizer_discrim = torch.optim.Adam(discrim.parameters(), lr = 0.001)
    bce_loss = torch.nn.BCELoss()

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

    #exit(0)

    # -------------------------------------
    #  and heeere we go
    # -------------------------------------
    training_losses = AverageMeter()
    auxiliary_data = []
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    ikk = 0
    for kk in optimizer.param_groups:
        if kk['lr'] > 0:
            ikk = kk
            break

    d_real_label = Variable(torch.ones(1,) - 0.1).cuda()
    d_fake_label = Variable(torch.zeros(1,) + 0.1).cuda()

    for t in range(args.numEpoch):
        print("The id of this in-training network is " + unique_id)

        #Turn into training mode
        model = model.train()

        for i, (X0_half, X1_half, y_half) in enumerate(train_loader):
            if i >= 10: 
                break

            X0_half = X0_half.cuda()
            X1_half = X1_half.cuda()
            y_half = y_half.cuda()

            X0 = Variable(X0_half, requires_grad= False)
            X1 = Variable(X1_half, requires_grad= False)
            y  = Variable(y_half,requires_grad= False)

            # --------------------------------------------
            #  first, train the interpolation network
            # --------------------------------------------
            optimizer.zero_grad()

            # comment out cycle stuff for now
            #X0_y = model(torch.stack((X0, y, y), dim = 0))
            #y_X1 = model(torch.stack((y, X1, X1), dim = 0))
            #y_est = model(torch.stack((X0_y, y, y_X1), dim = 0))

            y_est = model(torch.stack((X0, y, X1), dim = 0))

            # pixel loss (MSE with epsilon -- they call it "Charbonnier loss")
            y_diff = y_est - y
            #pixel_loss = torch.mean(torch.abs(y_diff))
            pixel_loss = torch.mean(torch.sqrt(y_diff * y_diff + args.epsilon * args.epsilon))

            # discriminator loss (we want it to be 1 = real)
            discrim_out = discrim(y_est)
            discrim_loss = bce_loss(discrim_out, d_real_label)

            total_loss = pixel_loss + 0.3 * discrim_loss

            total_loss.backward()
            optimizer.step()

            # --------------------------------------------
            #  then, train the discriminator
            # --------------------------------------------
            for real_img in [X0, y, X1]:
                optimizer_discrim.zero_grad()

                discrim_real_out = discrim(real_img)
                discrim_real_loss = bce_loss(discrim_real_out, d_real_label)

                discrim_fake_out = discrim(y_est.detach())
                discrim_fake_loss = bce_loss(discrim_fake_out, d_fake_label)

                discrim_total_loss = 0.5 * discrim_real_loss + 0.5 * discrim_fake_loss
                discrim_total_loss.backward()

                optimizer_discrim.step()

            # --------------------------------------------
            #  finally, output some stuff
            # --------------------------------------------
            training_losses.update(total_loss.item(), BATCH_SIZE)
            if i % max(1, int(int(len(train_set) / BATCH_SIZE) / 500.0)) == 0:

                print("Ep [" + str(t) +"/" + str(i) +
                                    "]\tl.r.: " + str(round(float(ikk['lr']),7))+
                                    "\tPix: " + str([round(pixel_loss.item(),5)]) +
                                    "\tFool: " + str([round(discrim_loss.item(),5)]) +
                                    "\tTotal: " + str([round(x.item(),5) for x in [total_loss]]) +
                                    "\tDiscrim: " + str([round(discrim_total_loss.item(),5)]) +
                                    "\tAvg. Loss: " + str([round(training_losses.avg, 5)]))

        if t == 1:
            # delete the pre validation weights for cleaner workspace
            if os.path.exists(args.save_path + "/epoch" + str(0) +".pth" ):
                os.remove(args.save_path + "/epoch" + str(0) +".pth")

        if os.path.exists(args.save_path + "/epoch" + str(t-1) +".pth"):
            os.remove(args.save_path + "/epoch" + str(t-1) +".pth")
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

                y_est = model(torch.stack((X0,y,X1),dim = 0))

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
