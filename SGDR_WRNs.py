"""
Lasagne implementation of SGDR on WRNs from "SGDR: Stochastic Gradient Descent with Restarts" (http://arxiv.org/abs/XXXX)
This code is based on Lasagne Recipes available at
https://github.com/Lasagne/Recipes/blob/master/papers/deep_residual_learning/Deep_Residual_Learning_CIFAR-10.py
and on WRNs implementation by Florian Muellerklein available at
https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee

"""

from __future__ import print_function

import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne
import math

from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer, batch_norm, BatchNormLayer
from lasagne.layers import ElemwiseSumLayer, NonlinearityLayer, GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.init import HeNormal
from lasagne.layers import Conv2DLayer as ConvLayer


# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(10000)

# ##################### Load data from CIFAR datasets #######################
# this code assumes the CIFAR dataset files have been extracted in current working directory
# from 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' for CIFAR-10
# from 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz' for CIFAR-100

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def load_data(dataset):
    xs = []
    ys = []
    if dataset == 'CIFAR-10':
        for j in range(5):
          d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
          x = d['data']
          y = d['labels']
          xs.append(x)
          ys.append(y)

        d = unpickle('cifar-10-batches-py/test_batch')
        xs.append(d['data'])
        ys.append(d['labels'])
    if dataset == 'CIFAR-100':
        d = unpickle('cifar-100-python/train')
        x = d['data']
        y = d['fine_labels']
        xs.append(x)
        ys.append(y)

        d = unpickle('cifar-100-python/test')
        xs.append(d['data'])
        ys.append(d['fine_labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

# ##################### Build the neural network model #######################



def ResNet_FullPre_Wide(input_var=None, nout=10,  n=3, k=2, dropoutrate = 0):
    '''
    Adapted from https://gist.github.com/FlorianMuellerklein/3d9ba175038a3f2e7de3794fa303f1ee
    which was tweaked to be consistent with 'Identity Mappings in Deep Residual Networks', Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    And 'Wide Residual Networks', Sergey Zagoruyko, Nikos Komodakis 2016 (http://arxiv.org/pdf/1605.07146v1.pdf)
    '''
    n_filters = {0:16, 1:16*k, 2:32*k, 3:64*k}

    # create a residual learning building block with two stacked 3x3 convlayers and dropout
    def residual_block(l, increase_dim=False, first=False, filters=16):
        if increase_dim:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        if first:
            # hacky solution to keep layers correct
            bn_pre_relu = l
        else:
            # contains the BN -> ReLU portion, steps 1 to 2
            bn_pre_conv = BatchNormLayer(l)
            bn_pre_relu = NonlinearityLayer(bn_pre_conv, rectify)

        # contains the weight -> BN -> ReLU portion, steps 3 to 5
        conv_1 = batch_norm(ConvLayer(bn_pre_relu, num_filters=filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=HeNormal(gain='relu')))

        if dropoutrate > 0:   # with dropout
            dropout = DropoutLayer(conv_1, p=dropoutrate)

            # contains the last weight portion, step 6
            conv_2 = ConvLayer(dropout, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=HeNormal(gain='relu'))
        else:   # without dropout
            conv_2 = ConvLayer(conv_1, num_filters=filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=HeNormal(gain='relu'))

        # add shortcut connections
        if increase_dim:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        elif first:
            # projection shortcut, as option B in paper
            projection = ConvLayer(l, num_filters=filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', b=None)
            block = ElemwiseSumLayer([conv_2, projection])
        else:
            block = ElemwiseSumLayer([conv_2, l])

        return block

    # Building the network
    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    # first layer=
    l = batch_norm(ConvLayer(l_in, num_filters=n_filters[0], filter_size=(3,3), stride=(1,1), nonlinearity=rectify, pad='same', W=HeNormal(gain='relu')))

    # first stack of residual blocks
    l = residual_block(l, first=True, filters=n_filters[1])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[1])

    # second stack of residual blocks
    l = residual_block(l, increase_dim=True, filters=n_filters[2])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[2])

    # third stack of residual blocks
    l = residual_block(l, increase_dim=True, filters=n_filters[3])
    for _ in range(1,n):
        l = residual_block(l, filters=n_filters[3])

    bn_post_conv = BatchNormLayer(l)
    bn_post_relu = NonlinearityLayer(bn_post_conv, rectify)

    # average pooling
    avg_pool = GlobalPoolLayer(bn_post_relu)

    # fully connected layer
    network = DenseLayer(avg_pool, num_units=nout, W=HeNormal(), nonlinearity=softmax)

    return network

# ############################# Batch iterator ###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False, augment=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if augment:
            # as in paper :
            # pad feature arrays with 4 pixels on each side
            # and do random cropping of 32x32
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]

        yield inp_exc, targets[excerpt]

# ############################## Main program ################################

def main(dataset = 'CIFAR-10', iscenario = 0, n=5, k = 1, num_epochs=82, model = None, irun = 0, Te = 2.0, E1 = 41, E2 = 61, E3 = 81,
         lr = 0.1, lr_fac = 0.1, reg_fac = 0.0005, t0 = math.pi/2.0, Estart = 0, dropoutrate = 0, multFactor = 1):
    # Check if CIFAR data exists
    if dataset == 'CIFAR-10':
        if not os.path.exists("./cifar-10-batches-py"):
            print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
            return
        nout = 10
    if dataset == 'CIFAR-100':
        if not os.path.exists("./cifar-100-python"):
            print("CIFAR-100 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
            return
        nout = 100
    # Load the dataset
    print("Loading data...")
    data = load_data(dataset)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = ResNet_FullPre_Wide(input_var, nout,  n, k, dropoutrate)
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))


    if model is None:
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        # add weight decay
        all_layers = lasagne.layers.get_all_layers(network)
        sh_reg_fac = theano.shared(lasagne.utils.floatX(reg_fac))
        l2_penalty = lasagne.regularization.regularize_layer_params(all_layers, lasagne.regularization.l2) * sh_reg_fac
        loss = loss + l2_penalty

        # Create update expressions for training
        # Stochastic Gradient Descent (SGD) with momentum
        params = lasagne.layers.get_all_params(network, trainable=True)
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.momentum(loss, params, learning_rate=sh_lr, momentum=0.9)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Create a loss expression for validation/testing
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)

    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # statistics file
    filename = "stat_{}_{}.txt".format(iscenario, irun)
    myfile=open(filename, 'w+', 0)
    start_time0 = time.time()

    tt = 0
    TeNext = Te
    batchsize = 128

    if model is None:
        # launch the training loop
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # shuffle training data
            train_indices = np.arange(100000)
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices,:,:,:]
            Y_train = Y_train[train_indices]

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()

            for batch in iterate_minibatches(X_train, Y_train, batchsize, shuffle=True, augment=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

                if (epoch+1 >= Estart): # time to start adjust learning tate
                    dt = 2.0*math.pi/float(2.0*Te)
                    tt = tt + float(dt)/(len(Y_train)/float(batchsize))
                    if tt >= math.pi:
                        tt = tt - math.pi
                    curT = t0 + tt
                    new_lr = lr * (1.0 + math.sin(curT))/2.0    # lr_min = 0, lr_max = lr
                    sh_lr.set_value(lasagne.utils.floatX(new_lr))

            if (epoch+1 == TeNext):     # time to restart
                tt = 0                  # by setting to 0 we set lr to lr_max, see above
                Te = Te * multFactor    # change the period of restarts
                TeNext = TeNext + Te    # note the next restart's epoch

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

            # print some statistics
            myfile.write("{}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\t{:.15g}\n".format(epoch, float(sh_lr.get_value()),
                    time.time() - start_time0, Te, train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100))

            # dump the network weights to a file :
            if epoch % 10 == 0:
                filesave = "network_{}_{}_{}.npz".format(iscenario,irun,epoch)
                np.savez(filesave, *lasagne.layers.get_all_param_values(network))

            # adjust learning rate as in the original approach
            if (epoch+1) == E1 or (epoch+1) == E2 or (epoch+1) == E3:
                new_lr = sh_lr.get_value() * lr_fac
                print("New LR:"+str(new_lr))
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

    else:
        # load network weights from model file
        with np.load(model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    myfile.close()

    # Calculate validation error of model:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


if __name__ == '__main__':

    # the only input is 'iscenario' index used to reproduce the experiments given in the paper
    # scenario #1 and #2 correspond to the original multi-step learning rate decay on CIFAR-10
    # scenarios [3-6] are 4 options for our SGDR
    # scenarios [7-10] are the same options but for 2 times wider WRNs, i.e., WRN-28-20
    # scenarios [11-20] are the same as [1-10] but for CIFAR-100
    iscenario = int(sys.argv[1])
    model = None

    dataset = 'CIFAR-10'

    iruns = [1,2,3,4,5]
    lr = 0.05
    lr_fac = 0.2
    reg_fac = 0.0005
    t0 = math.pi/2.0
    Te = 50
    dropoutrate = 0
    multFactor = 1
    num_epochs = 200
    E1 = -1;    E2 = -1;     E3 = -1;   Estart = -1

    # CIFAR-10
    if (iscenario == 1):    dataset = 'CIFAR-10';     n = 4;  k = 10;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.1;
    if (iscenario == 2):    dataset = 'CIFAR-10';     n = 4;  k = 10;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.05;
    if (iscenario == 3):    dataset = 'CIFAR-10';     n = 4;  k = 10;  Te = 50;
    if (iscenario == 4):    dataset = 'CIFAR-10';     n = 4;  k = 10;  Te = 100;
    if (iscenario == 5):    dataset = 'CIFAR-10';     n = 4;  k = 10;  Te = 1;      multFactor = 2;
    if (iscenario == 6):    dataset = 'CIFAR-10';     n = 4;  k = 10;  Te = 10;     multFactor = 2;
    if (iscenario == 7):    dataset = 'CIFAR-10';     n = 4;  k = 20;  Te = 50;
    if (iscenario == 8):    dataset = 'CIFAR-10';     n = 4;  k = 20;  Te = 100;
    if (iscenario == 9):    dataset = 'CIFAR-10';     n = 4;  k = 20;  Te = 1;      multFactor = 2;
    if (iscenario == 10):   dataset = 'CIFAR-10';     n = 4;  k = 20;  Te = 10;     multFactor = 2;

    # the same for CIFAR-100
    if (iscenario == 11):   dataset = 'CIFAR-100';    n = 4;  k = 10;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.1;
    if (iscenario == 12):   dataset = 'CIFAR-100';    n = 4;  k = 10;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.05;
    if (iscenario == 13):   dataset = 'CIFAR-100';    n = 4;  k = 10;  Te = 50;
    if (iscenario == 14):   dataset = 'CIFAR-100';    n = 4;  k = 10;  Te = 100;
    if (iscenario == 15):   dataset = 'CIFAR-100';    n = 4;  k = 10;  Te = 1;      multFactor = 2;
    if (iscenario == 16):   dataset = 'CIFAR-100';    n = 4;  k = 10;  Te = 10;     multFactor = 2;
    if (iscenario == 17):   dataset = 'CIFAR-100';    n = 4;  k = 20;  Te = 50;
    if (iscenario == 18):   dataset = 'CIFAR-100';    n = 4;  k = 20;  Te = 100;
    if (iscenario == 19):   dataset = 'CIFAR-100';    n = 4;  k = 20;  Te = 1;      multFactor = 2;
    if (iscenario == 20):   dataset = 'CIFAR-100';    n = 4;  k = 20;  Te = 10;     multFactor = 2;

    # very wide nets on CIFAR-10 and CIFAR-100
    if (iscenario == 21):   dataset = 'CIFAR-10';     n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.1;
    if (iscenario == 22):   dataset = 'CIFAR-10';     n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.05;
    if (iscenario == 23):   dataset = 'CIFAR-100';    n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.1;
    if (iscenario == 24):   dataset = 'CIFAR-100';    n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.05;
    if (iscenario == 25):   dataset = 'CIFAR-10';     n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.1;
    if (iscenario == 26):   dataset = 'CIFAR-10';     n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.05;
    if (iscenario == 27):   dataset = 'CIFAR-100';    n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.1;
    if (iscenario == 28):   dataset = 'CIFAR-100';    n = 4;  k = 20;  E1 = 60;    E2 = 120;    E3 = 160;  Estart = 10000;   lr = 0.05;

    for irun in iruns:
        main(dataset, iscenario, n, k, num_epochs, model, irun, Te, E1, E2, E3, lr, lr_fac, reg_fac, t0, Estart, dropoutrate, multFactor)