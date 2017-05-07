#!/usr/bin/env python

import math

if __name__ == '__main__':
    # minimum dimension of input image
    min_dim = 300
    # scale_1 ==> 40 x 40
    # scale_2 ==> 20 x 20
    # scale_3 ==> 10 x 10
    # scale_4 ==> 5 x 5
    # scale_5 ==> 3 x 3
    mbox_source_layers = ['scale_1', 'scale_2', 'scale_3', 'scale_4', 'scale_5']
    mbox_source_layers_size = [40, 20, 10, 5, 3]
    # in percent %
    min_ratio = 20
    max_ratio = 90

    # calculate min_size and max_size
    step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)

    min_sizes = [min_dim * 10 / 100.] + min_sizes
    max_sizes = [min_dim * 20 / 100.] + max_sizes

    # calculate steps
    steps = [int(math.ceil(float(min_dim) / x)) for x in mbox_source_layers_size]

    print '====== Prior Box Param ======'
    print 'min_raio: {}'.format(min_ratio)
    print 'max_ratio: {}'.format(max_ratio)
    print '-' * 48
    for i in xrange(len(mbox_source_layers)):
        print '[{}] [{:2d} x {:2d}] min_size={:.1f}, max_size={:.1f}, step={}'.format(
            mbox_source_layers[i], mbox_source_layers_size[i], mbox_source_layers_size[i],
            min_sizes[i], max_sizes[i], steps[i])
    print '-' * 48
