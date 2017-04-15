import caffe
import os,sys
sys.path.append('/home/sunxm/fcn.berkeleyvision.org')
import surgery, score

import numpy as np


try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/alexnet-fcn.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#surgery.interp(solver.net, interp_layers)
base_net = caffe.Net('caffenet.prototxt', weights, caffe.TEST)
surgery.transplant(solver.net, base_net)
del base_net

# scoring
val = np.loadtxt('../data/pascal/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
