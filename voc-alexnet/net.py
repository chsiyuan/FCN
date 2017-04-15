import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, group = 1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,group=group,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=3, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def lrn(bottom, local_size=5, alpha=1e-4, beta=0.75):
    return L.LRN(bottom, local_size=5, alpha=1e-4, beta=0.75)

def fcn(split):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            seed=1337)
    if split == 'train':
        pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
        pylayer = 'SBDDSegDataLayer'
    else:
        pydata_params['voc_dir'] = '../data/pascal/VOC2011'
        pylayer = 'VOCSegDataLayer'
    n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1, n.relu1 = conv_relu(n.data,96,ks=11,stride=4,pad=100)
    n.pool1 = max_pool(n.relu1)

    n.lrn1 = lrn(n.pool1)

    n.conv2, n.relu2 = conv_relu(n.lrn1, 256, ks=5, stride=1, pad=2, group=2)
    n.pool2 = max_pool(n.relu2)

    n.lrn2 = lrn(n.pool2)

    n.conv3, n.relu3 = conv_relu(n.lrn2, 384, ks=3, stride = 1, pad=1)
    n.conv4, n.relu4 = conv_relu(n.relu3, 384, ks=3, stride = 1, pad=1, group=2)
    n.conv5, n.relu5 = conv_relu(n.relu4, 256, ks=3, stride = 1, pad=1, group=2)
    n.pool5 = max_pool(n.relu5)


    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=6, stride = 1, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, stride = 1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.score_fr = L.Convolution(n.drop7, num_output=21, kernel_size=1, pad=0, stride = 1,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=63, stride=32,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=True, ignore_label=255))

    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('seg11valid')))

if __name__ == '__main__':
    make_net()
