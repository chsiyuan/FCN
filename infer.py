import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('data/pascal/VOC2011/JPEGImages/2007_000033.jpg')
#im = Image.open('data/pascal/bicycle.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
# net = caffe.Net('voc-fcn-alexnet/deploy.prototxt', 'ilsvrc-nets/caffenet-fcn32.caffemodel', caffe.TEST)
net = caffe.Net('voc-fcn-caffe16s-back/deploy.prototxt','voc-fcn-caffe16s-back/snapshot/train_iter_76000.caffemodel', caffe.TEST)
#net = caffe.Net('voc-fcn16s/deploy.prototxt','ilsvrc-nets/fcn16s.caffemodel', caffe.TEST)
#net = caffe.Net('voc-fcn4s/deploy.prototxt','voc-fcn4s/snapshot/train_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

plt.imsave('test3_caffe16s_76000.png',out,cmap='gray')
