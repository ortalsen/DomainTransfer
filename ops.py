import tensorflow as tf
import numpy as np
from scipy.io import loadmat, savemat
import os

def BNDAlayer(inputs, a, mode = 'train',domain='source'):
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    if mode == 'train':
        input_source, input_target = tf.split(inputs,num_or_size_splits=2,axis=0)
        mu_src, var_src = tf.nn.moments(input_source,[0,1,2]) # maybe should be tf.nn.moments(input_source,[0])
        mu_trgt, var_trgt = tf.nn.moments(input_target,[0,1,2])
        maintain_stats_avg = ema.apply([mu_src,var_src,mu_trgt,var_trgt])
        stats_avg = [ema.average(mu_src),ema.average(var_src),ema.average(mu_trgt),ema.average(var_trgt)]
        mu_sta = tf.scalar_mul(a,mu_src)+tf.scalar_mul((1.0-a),mu_trgt)
        var_sta = tf.scalar_mul(a,var_src+tf.square(mu_src))+\
                  tf.scalar_mul((1.0-a),var_trgt + tf.square(mu_trgt)) - tf.square(mu_sta)
        mu_tsa = tf.scalar_mul(a,mu_trgt)+tf.scalar_mul((1.0-a),mu_src)
        var_tsa = tf.scalar_mul(a,var_trgt+tf.square(mu_trgt))+\
                  tf.scalar_mul((1.0-a),var_src + tf.square(mu_src)) - tf.square(mu_tsa)
        input_source = tf.nn.batch_normalization(input_source,mu_sta,var_sta, offset=None,scale=None,variance_epsilon=1e-6)
        input_target = tf.nn.batch_normalization(input_target,mu_tsa,var_tsa, offset=None,scale=None,variance_epsilon=1e-6)
        output = tf.concat([input_source,input_target],axis=0)
    else:
        mu_src, var_src = tf.nn.moments(inputs, [0, 1, 2])  # maybe should be tf.nn.moments(input_source,[0])
        mu_trgt, var_trgt = tf.nn.moments(inputs, [0, 1, 2])
        ema.apply([mu_src, var_src, mu_trgt, var_trgt])
        stats_avg = [ema.average(mu_src), ema.average(var_src), ema.average(mu_trgt), ema.average(var_trgt)]
        '''MU_SRC = tf.placeholder(tf.float32, shape=mu_src._shape_as_list(),name='MU_SRC')
        VAR_SRC = tf.placeholder(tf.float32,shape=var_src._shape_as_list(),name='VAR_SRC')
        MU_TRG = tf.placeholder(tf.float32,shape=mu_trgt._shape_as_list(),name='MU_TRG')
        VAR_TRG = tf.placeholder(tf.float32,shape=var_trgt._shape_as_list(),name='VAR_TRG')'''
        if domain == 'source':
            output = tf.nn.batch_normalization(inputs, ema.average(mu_src), ema.average(var_src), offset=None, scale=None,variance_epsilon=1e-6)
        else:
            output = tf.nn.batch_normalization(inputs, ema.average(mu_trgt), ema.average(var_trgt), offset=None, scale=None, variance_epsilon=1e-6)
        maintain_stats_avg = None
        #stats_avg = None
    return output, maintain_stats_avg, stats_avg

def quantize_image(inputs, factor, minVal, maxVal):
    inputs[inputs<minVal] = minVal
    inputs[inputs>maxVal] = maxVal
    inputs = np.round(inputs/factor)
    inputs = inputs-inputs.min()
    return inputs


def image_2_1hot(inputs, levels):
    shape = inputs._shape_as_list()
    output = tf.reshape(inputs,shape=[shape[0]*shape[1]*shape[2],shape[3]])
    output = tf.one_hot(output,depth=levels)
    return output

def prob_2_image(probs_input):
    images = np.argmax(probs_input, 3)
    return images

def entropy_loss(input):
    loss = tf.reduce_mean(tf.reduce_sum(-tf.multiply(tf.clip_by_value(input, 1e-7, 1.0),tf.log(tf.clip_by_value(input, 1e-7, 1.0))), axis=1))
    return loss

def load_data(path,samples, shape, labels=False):
    data = np.zeros(shape)
    i = 0
    for sample in samples:
        mat = loadmat(path+'/'+sample)
        ind = np.argmin([str.find(key, '_') for key in mat.keys()])
        im = mat[mat.keys()[ind]]
        if labels:
            if im.max()<=1:
                im = im-im.min()
                im = im/im.max()
                im = im*360 + 40
        im = np.expand_dims(np.expand_dims(im,0),3)
        data[i,:,:,:] = im
        i+=1
    return data

def save_image(prob_images_batch, path, image_names):
    values_images_batch = prob_2_image(prob_images_batch)
    for ii in range(values_images_batch.shape[0]):
        im = values_images_batch[ii,:,:].squeeze()
        save_path = os.path.join(path,'CT_'+image_names[ii])
        im_dict = {'env':im}
        savemat(save_path,mdict=im_dict)