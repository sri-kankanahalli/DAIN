import time
import os
from torch.autograd import Variable
import math
import torch

import random
import numpy as np
import numpy
import networks
from datasets import pixel_triplets_dataset

from scipy.misc import imread, imsave
from AverageMeter import  *

from PIL import Image, ImageChops

def np_to_pil_image(im):
    im = im.clip(0.0, 1.0) * 255.0
    im = im.transpose((1, 2, 0))
    im = np.round(im).astype(np.uint8)

    return Image.fromarray(im)

torch.backends.cudnn.benchmark = True

model = networks.DAIN(channel = 3,
                      filter_size = 4,
                      timestep = 0.5,
                      training = False)

model = model.cuda()

WEIGHTS_PATH = './model_weights/epoch0.pth'

if os.path.exists(WEIGHTS_PATH):
    pretrained_dict = torch.load(WEIGHTS_PATH)

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We didn't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval()

dataset_dir = "./pixel_triplets"

random.seed(1337)

img_list = pixel_triplets_dataset.load_text_file(dataset_dir, "tri_trainlist.txt")

#for i in range(0, len(img_list)):
#    if 'captain' in img_list[i]:
#        print(i, img_list[i])
#exit(0)

# 7 is captain commando
# 172 is ken

# 67 train is a girl walking
# 23 train is iron man walk

img_path = img_list[5]

X0, X2, y = pixel_triplets_dataset.pixel_triplets_loader(dataset_dir, img_path, data_aug = False)

torch.set_grad_enabled(False)

X0_t = torch.from_numpy(X0).type(torch.cuda.FloatTensor)
y_t  = torch.from_numpy(y).type(torch.cuda.FloatTensor)
X2_t = torch.from_numpy(X2).type(torch.cuda.FloatTensor)

X0_t = Variable(torch.unsqueeze(X0_t,0))
y_t  = Variable(torch.unsqueeze(y_t,0))
X2_t = Variable(torch.unsqueeze(X2_t,0))

print("X0:", X0_t.shape)
print("y: ", y_t.shape)
print("X2:", X2_t.shape)



X0_y_p, _, _ = model(torch.stack((X0_t, y_t), dim = 0))
X0_y_p = X0_y_p[1]
print("X0_y_p:", X0_y_p.shape)

y_X2_p, _, _ = model(torch.stack((y_t, X2_t), dim = 0))
y_X2_p = y_X2_p[1]
print("y_X2_p:", y_X2_p.shape)

y_p,    _, _ = model(torch.stack((X0_y_p, y_X2_p), dim = 0))
y_p = y_p[1]
print("y_p:", y_p.shape)

X0_y_p = X0_y_p.data.cpu().numpy()[0]
y_p = y_p.data.cpu().numpy()[0]
y_X2_p = y_X2_p.data.cpu().numpy()[0]


X0  = np_to_pil_image(X0)
X0_y_p = np_to_pil_image(X0_y_p)
y   = np_to_pil_image(y)
y_p = np_to_pil_image(y_p)
y_X2_p = np_to_pil_image(y_X2_p)
X2  = np_to_pil_image(X2)

canvas = Image.new('RGB', (X0.size[0] * 5, X0.size[1] * 2), (255, 0, 255))

canvas.paste(X0, (0,              0))
canvas.paste(X0, (X0.size[0],     0))
canvas.paste(y,  (X0.size[0] * 2, 0))
canvas.paste(X2, (X0.size[0] * 3, 0))
canvas.paste(X2, (X0.size[0] * 4, 0))

canvas.paste(X0,      (0,              X0.size[1]))
canvas.paste(X0_y_p,  (X0.size[0]    , X0.size[1]))
canvas.paste(y_p,     (X0.size[0] * 2, X0.size[1]))
canvas.paste(y_X2_p,  (X0.size[0] * 3, X0.size[1]))
canvas.paste(X2,      (X0.size[0] * 4, X0.size[1]))

canvas.save("canvas.png")
