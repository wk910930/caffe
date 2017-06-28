#!/usr/bin/env python
"""
Have a view of net
"""

import argparse

from google.protobuf import text_format

import caffe
from caffe.proto import caffe_pb2

def parse_args():
    """ parse input arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        required=True, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='binary caffemodel',
                        required=False, type=str)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with args:'
    print args

    caffe.set_mode_cpu()
    net = caffe.Net(args.prototxt, caffe.TEST)

    # PROTOTXT
    print '== NET =='
    net_proto = caffe_pb2.NetParameter()
    text_format.Merge(open(args.prototxt).read(), net_proto)
    print net_proto

    # ACTIVATIONS
    # For each layer, check the activation shapes, which typically
    # have the form (batch_size, channel_dim, height, width).
    # The activations are exposed as an OrderedDict, net.blobs.
    print '== Activations =='
    for layer_name, blob in net.blobs.iteritems():
        print layer_name + '\t' + str(blob.data.shape)

    # PARAMETERS
    # The parameters are exposed as another OrderedDict, net.params.
    # We need to index the resulting values with either [0] for weights or [1] for biases.
    # The param shapes typically have the
    # form (output_channels, input_channels, filter_height, filter_width) (for the weights)
    # and the 1-dimensional shape (output_channels,) (for the biases).
    print '== Parameters =='
    for layer_name, param in net.params.iteritems():
        item = layer_name + '\t'
        for i in xrange(len(param)):
            item = item + str(param[i].data.shape) + ' '
        print item
