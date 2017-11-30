import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from ops import BNDAlayer



class MultiRes (object):
    def __init__(self, args):
        self.DS_FACTOR = args.DS_FACTOR
        self.kernel_size = [args.kernel_size, args.kernel_size]
        self.input_width = args.input_width
        self.MAX_BIF = args.MAX_BIF
        self.bif_num = 0
        self.mode = args.phase
        self.stride = [1, 1, 1, 1]
        self.nninit = tf.contrib.layers.variance_scaling_initializer()
        self.nninit_bias = tf.constant_initializer(0.0, dtype=tf.float32)
        self.active = tf.nn.relu
        self.labels = round((args.maxVal - args.minVal) / args.QuanFactor) + 1.0
        self.initial_depth = args.initial_depth
        self.input_depth = args.input_ch
        self.stats_avg = []


    def _make_full_model(self,initial_input, domain='source'):
        self.domain = domain
        initial_depth = self.initial_depth
        input_depth = self.input_depth
        input_width = self.input_width
        conv1 = slim.conv2d(initial_input,num_outputs=initial_depth,kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
        with tf.variable_scope("alpha"):
            a1 = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=1.0, maxval=1.0))
            a1 = tf.clip_by_value(a1,0.5,1.0)
        conv1, self.moving_stats_avg, stats_avg1 = BNDAlayer(conv1,a1,mode=self.mode, domain=self.domain)
        self.stats_avg.append(stats_avg1)
        conv1 = self.active(conv1)
        conv1 = slim.avg_pool2d(conv1,[self.DS_FACTOR,self.DS_FACTOR])
        sub_branch = self.make_partial_tree(initial_depth,conv1,input_width)
        input_shape = sub_branch._shape_as_list()
        size = np.array([input_shape[1] * 2, input_shape[2] * 2], dtype=np.float32)
        lp_branch = tf.image.resize_images(sub_branch, size=size, method=1)
        lp_branch = slim.conv2d(lp_branch, num_outputs=1, kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
        with tf.variable_scope("alpha"):
            a2 = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=1.0, maxval=1.0))
            a2 = tf.clip_by_value(a2, 0.5, 1.0)
        lp_branch, moving_stats_avg2, stats_avg2 = BNDAlayer(lp_branch, a2, mode=self.mode, domain=self.domain)
        if self.mode == 'train':
            self.moving_stats_avg = tf.group(self.moving_stats_avg,moving_stats_avg2)
        self.stats_avg.append(stats_avg2)
        lp_branch = self.active(lp_branch)
        hp_branch = self.make_hp_branch(input_depth,initial_depth,initial_input)
        concat_branch = tf.concat(axis=3,values=[lp_branch,initial_input,hp_branch])
        conv_last = slim.conv2d(concat_branch, num_outputs=self.labels, kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
        return conv_last, self.moving_stats_avg, self.stats_avg


    def make_partial_tree(self,input_depth, input_branch,width):
        self.bif_num += 1
        print('Bifurcation number')
        print(self.bif_num)
        print('partial tree input depth')
        print(input_depth)
        print('partial tree input width')
        print(width)
        if(input_depth<=1 or self.bif_num>self.MAX_BIF):
           print('max depth')
           return input_branch
        else:
            output_depth = 2*input_depth
            conv_layer = slim.conv2d(input_branch, num_outputs=output_depth,kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
            with tf.variable_scope("alpha"):
                a1 = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=1.0, maxval=1.0))
                a1 = tf.clip_by_value(a1, 0.5, 1.0)
            conv_layer, moving_stats_avg1, stats_avg1 = BNDAlayer(conv_layer, a1, mode=self.mode, domain=self.domain)
            conv_layer = self.active(conv_layer)
            conv_layer = slim.avg_pool2d(conv_layer, [self.DS_FACTOR, self.DS_FACTOR])
            width = width/self.DS_FACTOR
            sub_branch = self.make_partial_tree(output_depth, conv_layer,width)
            input_shape = sub_branch._shape_as_list()
            size = np.array([input_shape[1]*2,input_shape[2]*2],dtype=np.float32)
            lp_branch = tf.image.resize_images(sub_branch,size=size,method=1)
            lp_branch = slim.conv2d(lp_branch, num_outputs=input_depth,kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
            with tf.variable_scope("alpha"):
                a2 = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=1.0, maxval=1.0))
                a2 = tf.clip_by_value(a2, 0.5, 1.0)
            lp_branch, moving_stats_avg2, stats_avg2 = BNDAlayer(lp_branch, a2, mode=self.mode, domain=self.domain)
            if self.mode=='train':
                self.moving_stats_avg = tf.group(self.moving_stats_avg, moving_stats_avg1, moving_stats_avg2)
            self.stats_avg.append(stats_avg1)
            self.stats_avg.append(stats_avg2)
            lp_branch = self.active(lp_branch)
            hp_branch = self.make_hp_branch(input_depth,output_depth,input_branch)
            concat_branch = tf.concat(axis=3,values=[lp_branch,input_branch,hp_branch])
            conv_last = slim.conv2d(concat_branch,num_outputs=input_depth,kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=self.active,biases_initializer=self.nninit_bias)
            return conv_last

    def make_hp_branch(self,input_depth,output_depth, input_branch):
        conv = slim.conv2d(input_branch, num_outputs=output_depth,kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
        with tf.variable_scope("alpha"):
            a1 = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=1.0, maxval=1.0))
            a1 = tf.clip_by_value(a1, 0.5, 1.0)
        conv, moving_stats_avg1, stats_avg1 = BNDAlayer(conv,a1,mode=self.mode, domain=self.domain)
        conv = self.active(conv)
        conv = slim.avg_pool2d(conv, [self.DS_FACTOR, self.DS_FACTOR])
        input_shape = conv._shape_as_list()
        size = np.array([input_shape[1] * 2, input_shape[2] * 2], dtype=np.float32)
        conv = tf.image.resize_images(conv, size=size, method=1)
        conv = slim.conv2d(conv, num_outputs=input_depth, kernel_size=self.kernel_size, weights_initializer=self.nninit,activation_fn=None,biases_initializer=self.nninit_bias)
        with tf.variable_scope("alpha"):
            a2 = tf.Variable(initial_value=tf.random_uniform(shape=[], minval=1.0, maxval=1.0))
            a2 = tf.clip_by_value(a2, 0.5, 1.0)
        conv, moving_stats_avg2, stats_avg2 = BNDAlayer(conv, a2, mode=self.mode, domain=self.domain)
        conv = self.active(conv)
        if self.mode=='train':
            self.moving_stats_avg = tf.group(self.moving_stats_avg, moving_stats_avg1, moving_stats_avg2)
        self.stats_avg.append(stats_avg1)
        self.stats_avg.append(stats_avg2)
        return conv