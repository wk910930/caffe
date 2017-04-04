# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on
top of the Caffe deep learning library.

Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk),
Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor:
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
"""

import sys
# Add caffe for searching python interface
sys.path.insert(0, '../../python')
import caffe
import numpy as np
from PIL import Image as PILImage

def getpallete(num_cls):
    # this function is to get the color-map for visualizing the segmentation mask
    n = num_cls
    pallete = [0]*(n*3)
    for j in xrange(0, n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while lab > 0:
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete

def crfasrnn_segmenter(model_file, pretrained_file, gpudevice, inputs):
    if gpudevice >= 0:
        #Do you have GPU device? NO GPU is -1!
        has_gpu = 1
        #which gpu device is available?
        gpu_device = gpudevice#assume the first gpu device is available, e.g. Titan X
    else:
        has_gpu = 0
    if has_gpu == 1:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(model_file, pretrained_file, caffe.TEST)
    input_ = np.zeros((len(inputs), 500, 500, inputs[0].shape[2]), dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = in_

    # Segment
    caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]], dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = in_.transpose((2, 0, 1))

    out = net.forward_all(**{net.inputs[0]: caffe_in})
    predictions = out[net.outputs[0]]

    return predictions.argmax(axis=1).astype(np.uint8)

def run_crfasrnn(inputfiles, outputfiles, gpudevice):
    MODEL_FILE = 'TVG_CRFRNN_new_deploy.prototxt'
    PRETRAINED = 'TVG_CRFRNN_COCO_VOC.caffemodel'
    imgs = []
    heights = []
    widths = []
    for inputfile in inputfiles:
        IMAGE_FILE = inputfile
        input_image = 255 * caffe.io.load_image(IMAGE_FILE)
        input_image = resizeImage(input_image)

        width = input_image.shape[0]
        height = input_image.shape[1]
        maxDim = max(width, height)

        image = PILImage.fromarray(np.uint8(input_image))
        image = np.array(image)

        pallete = getpallete(256)

        mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        reshaped_mean_vec = mean_vec.reshape(1, 1, 3)

        # Rearrange channels to form BGR
        im = image[:, :, ::-1]
        # Subtract mean
        im = im - reshaped_mean_vec

        # Pad as necessary
        cur_h, cur_w, cur_c = im.shape
        heights.append(cur_h)
        widths.append(cur_w)
        pad_h = 500 - cur_h
        pad_w = 500 - cur_w
        im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)),
                    mode='constant', constant_values=0)
        imgs.append(im)

    # Get predictions
    segmentations = crfasrnn_segmenter(MODEL_FILE, PRETRAINED, gpudevice, imgs)

    # Write generated images
    for i in xrange(segmentations.shape[0]):
        cur_h = heights[i]
        cur_w = widths[i]
        segmentation2 = segmentations[i][0:cur_h, 0:cur_w]
        output_im = PILImage.fromarray(segmentation2)
        output_im.putpalette(pallete)
        output_im.save(outputfiles[i])

def resizeImage(image):
    width = image.shape[0]
    height = image.shape[1]
    maxDim = max(width, height)
    if maxDim > 500:
        if height > width:
            ratio = float(500.0 / height)
        else:
            ratio = float(500.0 / width)
        image = PILImage.fromarray(np.uint8(image))
        image = image.resize((int(height*ratio), int(width*ratio)), resample=PILImage.BILINEAR)
        image = np.array(image)
    return image

def main():
    inputfiles = ['input_1.jpg', 'input_2.jpg']
    outputfiles = ['output_1.png', 'output_2.png']
    assert len(inputfiles) == len(outputfiles), 'input and output file names do not match.'
    gpu_device = -1 # use -1 to run only on CPU, use others to run on GPU

    print 'Input files are {}'.format(inputfiles)
    print 'Output files are {}'.format(outputfiles)
    print 'GPU_DEVICE is {}'.format(gpu_device)

    run_crfasrnn(inputfiles, outputfiles, gpu_device)

if __name__ == "__main__":
    main()
