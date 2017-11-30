import os
from ops import *
from module import *
import datetime
import tensorflow as tf
import numpy as np
from random import shuffle

class AutoDIAL(object):
    def __init__(self,sess,args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.input_width = args.input_width
        self.val_iter = args.val_iter
        self.epoch = args.epoch
        self.dataset_dir = args.dataset_dir
        self.network = MultiRes(args)
        self.input_ch = args.input_ch
        self.minVal = args.minVal
        self.maxVal = args.maxVal
        self.QuanFactor = args.QuanFactor
        self.labels = int(round((self.maxVal-self.minVal)/self.QuanFactor)+1.0)
        self.Lambda = args.Lambda
        self.lr = args.lr
        self.checkpoint_dir = args.checkpoint_dir
        self.exp_name = args.exp_name
        self.continue_training = args.continue_training
        self.test_dir = args.test_dir
        self.domain = args.domain
        self.phase = args.phase

        self._build_model()
        self.saver = tf.train.Saver() # see how to save averages

    def _build_model(self):
        network = self.network
        if self.phase == 'train':
            self.input = tf.placeholder(tf.float32, shape=[2*self.batch_size, self.input_width, self.input_width, self.input_ch])
        else:
            self.input = tf.placeholder(tf.float32,shape=[None, self.input_width, self.input_width, self.input_ch])
        self.source_labels = tf.placeholder(tf.float32,shape=[self.batch_size,self.input_width,self.input_width,self.input_ch])
        self.output, self.moving_stats_avg, self.stats_avg = network._make_full_model(self.input)
        self.source_logits_images = tf.reshape(self.output[0:self.batch_size,:,:,:],[self.batch_size*self.input_width*self.input_width,self.labels])
        self.source_probs_images = tf.nn.softmax(self.output[0:self.batch_size,:,:,:])
        self.source_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=image_2_1hot(tf.cast(self.source_labels,dtype=tf.int32) , levels=self.labels),
                                                                   logits=self.source_logits_images))
        self.target_logits_images = tf.reshape(self.output[self.batch_size:, :, :, :],[self.batch_size*self.input_width*self.input_width,self.labels])
        self.target_probs_images = tf.nn.softmax(self.output[self.batch_size:,:,:,:])
        self.target_loss = entropy_loss(tf.nn.softmax(self.target_logits_images))
        self.AutoDIAL_loss = self.source_loss+self.Lambda*self.target_loss
        self.train_src_loss_summ = tf.summary.scalar('train_source_xentropy_loss', self.source_loss)
        self.train_trg_loss_summ = tf.summary.scalar('train_target_entropy_loss', self.target_loss)
        self.train_AD_loss_summ = tf.summary.scalar('train_AutoDIAL_loss', self.AutoDIAL_loss)
        '''self.train_src_loss_summ = tf.summary.scalar('source_xentropy_loss', self.source_loss)
        self.train_trg_loss_summ = tf.summary.scalar('target_entropy_loss', self.target_loss)
        self.train_AD_loss_summ = tf.summary.scalar('AutoDIAL_loss', self.AutoDIAL_loss)'''
        self.mean_val_AD_loss = tf.placeholder(tf.float32, shape=[])
        self.mean_val_src_loss = tf.placeholder(tf.float32, shape=[])
        self.mean_val_trg_loss = tf.placeholder(tf.float32, shape=[])
        self.val_src_loss_summ = tf.summary.scalar('validation_source_xentropy_loss', self.mean_val_AD_loss)
        self.val_trg_loss_summ = tf.summary.scalar('validation_target_entropy_loss', self.mean_val_trg_loss)
        self.val_AD_loss_summ = tf.summary.scalar('validation_AutoDIAL_loss', self.mean_val_src_loss)
        '''self.val_src_loss_summ = tf.summary.scalar('source_xentropy_loss', self.mean_val_AD_loss)
        self.val_trg_loss_summ = tf.summary.scalar('target_entropy_loss', self.mean_val_trg_loss)
        self.val_AD_loss_summ = tf.summary.scalar('AutoDIAL_loss', self.mean_val_src_loss)'''

    def train(self):
        with tf.control_dependencies([self.moving_stats_avg]):
            optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.AutoDIAL_loss)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        '''self.writer_train = tf.summary.FileWriter("./logs/train", self.sess.graph)
        self.writer_val = tf.summary.FileWriter("./logs/val", self.sess.graph)'''
        list_source = os.listdir('datasets/'+self.dataset_dir+'/source/samples')
        list_target = os.listdir('datasets/'+self.dataset_dir+'/target/samples')
        num_src_samples = list_source.__len__()
        num_trg_samples = list_target.__len__()
        num_src_train = round(num_src_samples*0.64)
        num_src_val = round(num_trg_samples*0.16)
        num_trg_train = round(num_trg_samples * 0.64)
        num_trg_val = round(num_trg_samples * 0.16)
        shuffle(list_source)
        shuffle(list_target)
        list_source_train = list_source[0:int(num_src_train)]
        list_source_val = list_source[int(num_src_train):int(num_src_train+num_src_val)]
        list_source_test = list_source[int(num_src_train+num_src_val):]
        list_target_train = list_target[0:int(num_trg_train)]
        list_target_val = list_target[int(num_trg_train):int(num_trg_train+num_trg_val)]
        list_target_test = list_target[int(num_trg_train+num_trg_val):]
        np.save('datasets/'+ self.dataset_dir + '/' + self.exp_name + '_list_source_test.npy',list_source_test)
        np.save('datasets/'+ self.dataset_dir + '/' + self.exp_name + '_list_target_test.npy',list_target_test)
        if self.continue_training and self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print 'starting training...'
        train_iter = 0
        for i in np.arange(self.epoch):
            j = 0
            val_iter = round(self.val_iter*(1.0/self.batch_size)*min(list_source_train.__len__(),list_target_train.__len__()))
            shuffle(list_source_train)
            shuffle(list_target_train)
            while j*self.batch_size+self.batch_size<=list_source_train.__len__() and j*self.batch_size+self.batch_size<=list_target_train.__len__():
                x_batch = np.zeros([2*self.batch_size,self.input_width,self.input_width,1])
                x_batch[0:self.batch_size,:,:,:] = load_data('datasets/'+self.dataset_dir+'/source/samples',list_source_train[j*self.batch_size:(j*self.batch_size+self.batch_size)]
                                                             ,shape=[self.batch_size,self.input_width,self.input_width,self.input_ch])
                x_batch[self.batch_size:,:,:,:] = load_data('datasets/'+self.dataset_dir+'/target/samples',list_target_train[j*self.batch_size:(j*self.batch_size+self.batch_size)]
                                                            , shape=[self.batch_size,self.input_width,self.input_width,self.input_ch])
                y_batch = load_data('datasets/'+self.dataset_dir+'/source/labels',list_source_train[j*self.batch_size:(j*self.batch_size+self.batch_size)]
                                    , shape=[self.batch_size,self.input_width,self.input_width,self.input_ch], labels=True)
                y_batch = quantize_image(y_batch,self.QuanFactor,self.minVal,self.maxVal)
                AD_loss_summ, src_loss_summ, trg_loss_summ, AD_loss  = self.sess.run([self.train_AD_loss_summ,self.train_src_loss_summ,self.train_trg_loss_summ, self.AutoDIAL_loss],
                                                              feed_dict={self.input: x_batch, self.source_labels: y_batch})
                '''self.writer_train.add_summary(AD_loss_summ, train_iter)
                self.writer_train.add_summary(src_loss_summ, train_iter)
                self.writer_train.add_summary(trg_loss_summ, train_iter)'''
                self.writer.add_summary(AD_loss_summ, train_iter)
                self.writer.add_summary(src_loss_summ, train_iter)
                self.writer.add_summary(trg_loss_summ, train_iter)
                now = datetime.datetime.now().ctime()
                print(now)
                msg = "Epoch: {0}, Minibatch: {1}, Training loss: {2}"
                print(msg.format(i , train_iter,  AD_loss))
                self.sess.run(optimizer, feed_dict={self.input: x_batch, self.source_labels: y_batch})
               # for a in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='alpha'):
                    #a = tf.clip_by_value(a,0.5,1.0)
                j+=1
                train_iter+=1
                if j % val_iter == 0:
                    k = 0
                    shuffle(list_source_val)
                    shuffle(list_target_val)
                    AD_loss = 0
                    SRC_loss = 0
                    TRG_loss = 0
                    while k * self.batch_size + self.batch_size <= list_source_val.__len__() and k * self.batch_size + self.batch_size <= list_target_val.__len__():
                        x_batch = np.zeros([2 * self.batch_size, self.input_width, self.input_width, 1])
                        x_batch[0:self.batch_size, :, :, :] = load_data('datasets/'+self.dataset_dir + '/source/samples',
                                                                        list_source_val[k * self.batch_size:(
                                                                        k * self.batch_size + self.batch_size)]
                                                                        , shape=[self.batch_size, self.input_width,
                                                                                 self.input_width, self.input_ch])
                        x_batch[self.batch_size:, :, :, :] = load_data('datasets/'+self.dataset_dir + '/target/samples',
                                                                       list_target_val[k * self.batch_size:(
                                                                       k * self.batch_size + self.batch_size)]
                                                                       , shape=[self.batch_size, self.input_width,
                                                                                self.input_width, self.input_ch])
                        y_batch = load_data('datasets/'+self.dataset_dir + '/source/labels', list_source_val[k * self.batch_size:(k * self.batch_size + self.batch_size)],
                                            shape=[self.batch_size, self.input_width, self.input_width, self.input_ch], labels=True)
                        y_batch = quantize_image(y_batch, self.QuanFactor, self.minVal, self.maxVal)
                        ad_loss, src_loss, trg_loss = self.sess.run([self.AutoDIAL_loss,self.source_loss,self.target_loss], feed_dict={self.input: x_batch, self.source_labels: y_batch})
                        AD_loss = AD_loss+ad_loss
                        SRC_loss = SRC_loss+src_loss
                        TRG_loss = TRG_loss+trg_loss
                        k += 1
                    mean_val_AD_loss = AD_loss/k
                    mean_val_src_loss = SRC_loss/k
                    mean_val_trg_loss = TRG_loss/k
                    msg = "mean loss on Validation-Set: {0}"
                    print(msg.format(mean_val_AD_loss))
                    val_src_loss_summ, val_trg_loss_summ, val_AD_loss_summ = self.sess.run([self.val_src_loss_summ,self.val_trg_loss_summ,self.val_AD_loss_summ],
                                                                                           feed_dict={self.mean_val_AD_loss:mean_val_AD_loss,
                                                                                                      self.mean_val_src_loss:mean_val_src_loss,
                                                                                                      self.mean_val_trg_loss:mean_val_trg_loss})
                    '''self.writer_val.add_summary(val_src_loss_summ,train_iter)
                    self.writer_val.add_summary(val_trg_loss_summ, train_iter)
                    self.writer_val.add_summary(val_AD_loss_summ, train_iter)'''
                    self.writer.add_summary(val_src_loss_summ, train_iter)
                    self.writer.add_summary(val_trg_loss_summ, train_iter)
                    self.writer.add_summary(val_AD_loss_summ, train_iter)
                    self.save(train_iter)


    def test(self):
        if self.domain == 'source':
            list_test = np.load('datasets/'+ self.dataset_dir + '/' + self.exp_name + '_list_source_test.npy')
        else:
            list_test = np.load('datasets/'+ self.dataset_dir + '/' + self.exp_name + '_list_target_test.npy')
        if not os.path.exists(self.test_dir+'/source'):
            os.makedirs(self.test_dir+'/source')
        if not os.path.exists(self.test_dir+'/target'):
            os.makedirs(self.test_dir+'/target')
        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        print 'starting evaluation...'
        k = 0
        #Loss = 0
        while k * self.batch_size + self.batch_size <= list_test.__len__() and k * self.batch_size + self.batch_size <= list_test.__len__():
            x_batch = load_data('datasets/' + self.dataset_dir + '/'+self.domain+'/samples',
                                                            list_test[k * self.batch_size:(
                                                                k * self.batch_size + self.batch_size)]
                                                            , shape=[self.batch_size, self.input_width,
                                                                     self.input_width, self.input_ch])
            #y_batch = load_data('datasets/' + self.dataset_dir + '/'+self.domain+'/labels',
                                #list_test[k * self.batch_size:(k * self.batch_size + self.batch_size)],
                                #shape=[self.batch_size, self.input_width, self.input_width, self.input_ch], labels=True)
            #y_batch = quantize_image(y_batch, self.QuanFactor, self.minVal, self.maxVal)
            #loss,probs_images = self.sess.run([self.source_loss,self.source_probs_images],
                                                        #feed_dict={self.input: x_batch, self.source_labels: y_batch})
            probs_images = self.sess.run(self.source_probs_images,feed_dict={self.input: x_batch})
            #Loss = Loss + loss
            save_image(probs_images,self.test_dir+'/'+self.domain,list_test[k * self.batch_size:(k * self.batch_size + self.batch_size)])
            k += 1
        #mean_test_loss = Loss / k
        #msg = "mean loss on " +self.domain+ " Test-Set: {0}"
        #print(msg.format(mean_test_loss))
        return 0

    def save(self, step):
        model_name = "AutoDIAL.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.exp_name)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.exp_name)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

