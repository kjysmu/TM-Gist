#!/usr/bin/env python




import time
import cPickle
import numpy as np
import theano
import theano.tensor as T
import lasagne
import sys

from lasagne.layers import *




def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x
    


def make_idx_data_cv(revs, word_idx_map, cv, max_l=56, k=300, filter_h=8):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train,Y_train,X_test,Y_test = [],[],[], []

    for rev in revs:
        if rev["y"]==3:
            rev["y"]=2        
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        
        if rev["split"] == cv:
            X_test.append(sent)
            Y_test.append(rev["y"])
        else:
            X_train.append(sent)
            Y_train.append(rev["y"])
            
            
    X_train = np.array([X_train], dtype="int")
    Y_train = np.array(Y_train, dtype="int")
    X_test = np.array([X_test], dtype="int")    
    Y_test = np.array(Y_test, dtype="int")  
    

    X_train=X_train.reshape(X_train.shape[1],1,max_l+2*filter_h-2)        
    X_test=X_test.reshape(X_test.shape[1],1,max_l+2*filter_h-2)       
    
    return X_train,Y_train,X_test,Y_test

def make_idx_data_TT(revs, word_idx_map, cv, max_l=72, k=300, filter_h=8):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train,Y_train,X_test,Y_test = [],[],[], []

    for rev in revs:
        if rev["y"]==3:
            rev["y"]=2        
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        
        if rev["y"] >= 10:
            X_test.append(sent)
            Y_test.append(rev["y"]-10)
        else:
            X_train.append(sent)
            Y_train.append(rev["y"])
            
            
    X_train = np.array([X_train], dtype="int")
    Y_train = np.array(Y_train, dtype="int")
    X_test = np.array([X_test], dtype="int")    
    Y_test = np.array(Y_test, dtype="int")  
    

    X_train=X_train.swapaxes(1,0)     
    X_test=X_test.swapaxes(1,0)        
    
    return X_train,Y_train,X_test,Y_test






# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    #excerpt = indices[start_idx + batchsize+1:]
    #yield inputs[excerpt], targets[excerpt]


    


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
if __name__ == '__main__': 
    
    #print 'watting'     
    #time.sleep(5 * 60)
    # Load the dataset
    print 'Loading data...'
    

    x = cPickle.load(open("F:/semEval_code/data/mr_4.p", "rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    #print "data loaded!"
    num_epochs=30
    max_l,k,filter_h1,n_filter,feat_num = 43, 300, 5, 400 , 100
    X_train,Y_train,X_test,Y_test = make_idx_data_TT(revs, word_idx_map, 0, max_l=max_l, k=k, filter_h=filter_h1)

    #X_train = X_train[:,0,:]
    #X_test = X_test[:,0,:]

    # Prepare Theano variables for inputs and targets
    input_varv = T.imatrix('inputs')
    input_var = T.itensor3('inputs')
    target_var = T.ivector('targets')
    
    a = np.zeros((300,1,1,300),dtype = 'float32')
    for i in range(300):
        a[i,0,0,i]=1
    
    k2 = 300

    network = lasagne.layers.InputLayer(shape=(None,1,  max_l+(filter_h1-1)*2 ),input_var=input_var)    
    network = EmbeddingLayer(network,input_size=W.shape[0],output_size=W.shape[1],W=W)
    
    '''
    #network.params[network.W].remove('trainable')
    #network = Conv2DLayer(network,num_filters= k2, filter_size=(1,300),W=a)
    network = Conv2DLayer(network,num_filters= k2, filter_size=(1,300))
    network = DimshuffleLayer(network,(0,3,2,1))
    '''
    
    network = lasagne.layers.reshape(network,([0],[2],[3]))
    network1= LSTMLayer(network, feat_num, nonlinearity = lasagne.nonlinearities.softplus, learn_init=True)
    network2= LSTMLayer(network, feat_num, nonlinearity = lasagne.nonlinearities.softplus, learn_init=True, backwards = True) 
    #network1= GRULayer(network, 100, learn_init=True)
    #network2= GRULayer(network, 100, learn_init=True, backwards = True)  
    network = concat([network,network1,network2],axis = 2)

    
    network = lasagne.layers.reshape(network,([0],1,[1],[2]))
    network = lasagne.layers.Conv2DLayer(
            batch_norm(network), num_filters = 300, filter_size=(filter_h1, 2*feat_num+k2),
            nonlinearity = lasagne.nonlinearities.softplus,
            W = lasagne.init.GlorotUniform())
    '''
    network = lasagne.layers.Conv2DLayer(
            batch_norm(network), num_filters = 200, filter_size=(filter_h1, 1),
            nonlinearity = lasagne.nonlinearities.softplus,
            W = lasagne.init.GlorotUniform())    
    '''
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=((max_l+filter_h1*2-10), 1))  
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=((max_l+filter_h1*2-6), 1))  

    '''
    network = lasagne.layers.DenseLayer(
        network,
        num_units=100,
        nonlinearity=lasagne.nonlinearities.softplus)
    '''
    network = lasagne.layers.DenseLayer(
            dropout(flatten(network)),
            num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid) 
    #lasagne.layers.batch_norm(network) instead of lasagne.layers.dropout(network, p=.5)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    
    l2_penalty = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2) * 1e-4
    loss = loss.mean()+l2_penalty
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(loss, params)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.binary_crossentropy(test_prediction,
                                                            target_var)
    test_l2_penalty = lasagne.regularization.regularize_layer_params(network, lasagne.regularization.l2) * 1e-4
    test_loss = test_loss.mean()+test_l2_penalty
    # As a bonus, also create an expression for the classification accuracy:
        
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
        
    #test_acc = T.mean(T.eq(test_prediction, target_var),dtype=theano.config.floatX)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    #for prediction and get score
    get_predd = theano.function([input_var], prediction)
    x1=X_test.shape[1]
    x2=X_test.shape[2]
    

    
    batch_n=100
    
    # Finally, launch the training loop.
    print 'Starting training...'
    # We iterate over epochs:
    print '\t','\t','epoch \t','time\t','train loss\t' ,'train acc\t','test loss\t','test acc','\t','score'
    for epoch in range(20):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        i=1
        for batch in iterate_minibatches(X_train, Y_train, batch_n, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            parameter = lasagne.layers.get_all_param_values(network)
            parameter[0][0] = np.array([0]*k)
            lasagne.layers.set_all_param_values(network,parameter)
            
            train_batches += 1
            sys.stdout.write("\r{0}/{1}".format(i*batch_n,len(X_train)))
            sys.stdout.flush()
            i+=1

        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_train, Y_train, 100, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1

        
       
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        
        
        for batch in iterate_minibatches(X_dev, Y_dev, 100, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        print '\t',epoch+1,'/',num_epochs,'\t',round(time.time() - start_time,1),'\t',round(train_err / train_batches,4),'\t\t',round(test_acc / test_batches,4),'\t\t',round(val_err / val_batches,4),'\t\t',round(val_acc / val_batches,4)
    
        if True:
            out = np.array([[0]])
            score_matrix=([[0.0,0.0],[0.0,0.0]])                
                     
            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 100, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
                out=np.append(out,np.array(get_predd(inputs)).argmax(-1))
            
            out=np.delete(out,0)# remove first low.. [0,0,0]
            y_pred = out
            for ii in range(len(y_pred)):
                score_matrix[Y_test[ii]][y_pred[ii]]+=1
                                   
            # pi ro should be changed

            ro_P = round(score_matrix[0][0]/(score_matrix[0][0]+score_matrix[0][1]+1E-5),4)    
            ro_N = round(score_matrix[1][1]/(score_matrix[1][1]+score_matrix[1][1]+1E-5),4)        
                
            print 'Final results:'
            print '  test loss:\t\t\t{:.6f}'.format(test_err / test_batches)
            print '  test accuracy:\t\t{:.2f} %'.format(
                test_acc / test_batches * 100) 
            print (ro_P+ro_N)/2,ro_P,ro_N
    



