import os
import os.path
import random
from scipy.misc import imread, imsave, imresize
from PIL import Image
from math import floor
import numpy as np
from .listdatasets import ListDataset

CANVAS_WIDTH = 192
CANVAS_HEIGHT = 128
BACKGROUND_COLOR = (255, 255, 255)

def load_text_file(root, fname, shuffle = True):
    text_path = os.path.join(root, fname)
    raw_im_list = open(text_path).read().splitlines()

    assert len(raw_im_list) > 0

    if (shuffle):
        random.shuffle(raw_im_list)

    return raw_im_list

def pixel_triplets(root, split = 1.0, single = False, task = 'interp'):
    train_list = load_text_file(root, "tri_trainlist.txt")
    val_list = load_text_file(root, "tri_vallist.txt")

    train_dataset = ListDataset(root, train_list, loader = pixel_triplets_loader)
    val_dataset = ListDataset(root, val_list, loader = pixel_triplets_loader)

    return train_dataset, val_dataset

def scale_image_to_canvas(im, scale):
    if (scale > 1):
        new_size = (im.size[0] * scale, im.size[1] * scale)
        im = im.resize(new_size, Image.NEAREST)

    canvas = Image.new('RGB', (CANVAS_WIDTH, CANVAS_HEIGHT), BACKGROUND_COLOR)
    canvas.paste(im, (CANVAS_WIDTH // 2 - im.size[0] // 2, CANVAS_HEIGHT // 2 - im.size[1] // 2))

    return canvas

def pixel_triplets_loader(root, im_path, data_aug = True):
    full_path = os.path.join(root, im_path)

    with open(full_path, 'rb') as f:
        # --------------------------------------------
        #  load image and split / augment the triplet as needed
        # --------------------------------------------
        combined_im = Image.open(f)
        imwidth = combined_im.size[0] // 3
        imheight = combined_im.size[1]

        im_before = combined_im.crop((0, 0, imwidth, imheight))
        im_mid    = combined_im.crop((imwidth, 0, imwidth * 2, imheight))
        im_after   = combined_im.crop((imwidth * 2, 0, imwidth * 3, imheight))

        '''
        if (data_aug):
            # flip image horizontally
            if (random.randint(0, 1)):
                im_before = im_before.transpose(Image.FLIP_LEFT_RIGHT)
                im_mid = im_mid.transpose(Image.FLIP_LEFT_RIGHT)
                im_after = im_after.transpose(Image.FLIP_LEFT_RIGHT)
        '''

        # --------------------------------------------
        #  nearest-neighbor upscale + pad to the correct canvas size
        # --------------------------------------------
        im_before = scale_image_to_canvas(im_before, 1)
        im_mid    = scale_image_to_canvas(im_mid, 1)
        im_after  = scale_image_to_canvas(im_after, 1)

        # --------------------------------------------
        #  finally, do all the technical Numpy conversion shit
        #      final output is 3xWxH
        # --------------------------------------------
        X0 = np.asarray(im_before).transpose((2, 0, 1))
        X2 = np.asarray(im_after).transpose((2, 0, 1))
        y  = np.asarray(im_mid).transpose((2, 0, 1))

        #X0[(X0 == (255, 0, 255)).all(axis = -1)] = (255, 255, 255)
        #X2[(X2 == (255, 0, 255)).all(axis = -1)] = (255, 255, 255)
        #y[(y == (255, 0, 255)).all(axis = -1)] = (255, 255, 255)

        X0 = X0.astype('float32') / 255.0
        X2 = X2.astype('float32') / 255.0
        y  =  y.astype('float32') / 255.0

        if (data_aug):
            # reverse order of triplets
            if (random.randint(0, 1)):
                tmp = X2
                X2 = X0
                X0 = tmp

        return X0, X2, y

