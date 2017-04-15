import caffe
import os,sys
sys.path.append('/home/chsiyuan/fcn.berkeleyvision.org')
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
base_net = caffe.Net('vgg16.prototxt', '../ilsvrc-nets/vgg16-fcn.caffemodel', caffe.TEST)
surgery.transplant(solver.net, base_net)
del base_net


# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/pascal/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
