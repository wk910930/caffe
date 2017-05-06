#!/usr/bin/env python

import math

if __name__ == '__main__':
    # minimum dimension of input image
    min_dim = 300
    # scale_1 ==> 38 x 38
    # scale_2 ==> 19 x 19
    # scale_3 ==> 10 x 10
    # scale_4 ==> 5 x 5
    # scale_5 ==> 3 x 3
    # scale_6 ==> 1 x 1
    mbox_source_layers = ['scale_1', 'scale_2', 'scale_3', 'scale_4', 'scale_5', 'scale_6']
    mbox_source_layers_size = [38, 19, 10, 5, 3, 1]
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
    steps = [8, 16, 32, 64, 100, 300]
    my_steps = [int(math.ceil(float(min_dim) / x)) for x in mbox_source_layers_size]

    print '====== Prior Box Param ======'
    print 'min_raio: {}'.format(min_ratio)
    print 'max_ratio: {}'.format(max_ratio)
    print '-' * 48
    for i in xrange(len(mbox_source_layers)):
        print '[{}] min_size={:.1f} max_size={:.1f}'.format(
            mbox_source_layers[i], min_sizes[i], max_sizes[i])
    print '-' * 48
    print 'steps={} (manually)'.format(steps)
    print 'steps={} (approximate)'.format(my_steps)
