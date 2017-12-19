from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf
from dataloader import dataloader
from ops import *
from utils import *
from glob import glob
import math
class Vid_Imagine(object):
  def __init__(self, sess,
         batch_size=64,
         num_epochs = 25,
         image_height=128, image_width=128, c_dim=1,

         conv_size = 9,
         sequence_len = 5,
         trans_par = 6,
         transformation='affine_transformation',

         dataset_name='CUF101Surfing',
         data_dir = '/ssd/10.10.20.21/share/guojiaming/UCF-101/Surfing/',
         feature = 'digits',

         z_dim=100,
         emcode_len = 512,
         clamp_lower=-0.01,clamp_upper=0.01,
         output_frames=4,
         video_len=5,
         is_flatten=False,
         is_conv=True,
         load_model=False):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: Dimension of dim for Z. [100]
      emcode_len: Dimension of encoded condition code [512]
      clamp_lower: clamp parameters in WGAN [-0.01]

      conv_size: convolution kernel size [9,16]
      sequence_len: transformation sequence length [5,10]
      trans_par: number of parameters in transformation [6,9*9,16*16]
      transformation: transformation model type [affine_transformation,conv_transformation]

      output_frames: number of frames reconstructed
      video_len: number of frames in imaginary video
      is_flatten: Flatten image as condition code 
      is_conv: Finetune alexnet or use custom conv

    """
    self.sess = sess
    # batch info
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    # input info
    self.image_height = image_height
    self.image_width = image_width
    self.c_dim = c_dim
    # dataset info
    self.dataset_name = dataset_name
    self.video_len = video_len
    self.data_dir = data_dir
    self.feature = feature
    # output info
    self.conv_size = conv_size
    self.trans_par = trans_par
    self.sequence_len = sequence_len
    self.output_frames = output_frames
    # parameter info
    self.is_flatten = is_flatten
    self.is_conv = is_conv
    self.transformation = transformation
    self.z_dim = z_dim
    self.emcode_len = emcode_len
    self.clamp_lower = clamp_lower
    self.clamp_upper = clamp_upper
    self.max_iter=500000
    self.load_model=load_model
    self.sampledir=self.dataset_name+'sample'+'conv'
    if not os.path.exists(self.sampledir):
        os.mkdir(self.sampledir)
    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.e_bn1 = batch_norm(name='e_bn1')
    self.e_bn2 = batch_norm(name='e_bn2')
    self.e_bn3 = batch_norm(name='e_bn3')
    self.e_bn4 = batch_norm(name='e_bn4')
    self.e_bn5 = batch_norm(name='e_bn5')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.build_model()

  def build_model(self):
    
    ###Sample Noise###
    z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')
    self.z = z

    ###Train data flow###
    self.images = tf.placeholder(tf.float32,[self.batch_size,self.video_len,self.image_height,self.image_width,self.c_dim],name='inputclip')
    stimage, real_video, stimage_64 = SequenceToImageAndVideo(self.images)
    self.imaginary = self.generator(z, stimage, stimage_64)
    
    ###Validate data flow###
    self.val_images = tf.placeholder(tf.float32,[self.batch_size,self.video_len,self.image_height,self.image_width,self.c_dim],name='varclip')
    val_stimage, val_real_video, val_stimage_64 = SequenceToImageAndVideo(self.val_images)
    self.samplers = self.generator(z, val_stimage, val_stimage_64,reuse=True)
    print self.samplers.shape
    ###Loss function###
    true_logit = self.VideoCritic(real_video)
    fake_logit = self.VideoCritic(self.imaginary,reuse = True)
    self.d_loss = -tf.reduce_mean(fake_logit - true_logit)
    self.g_loss = -tf.reduce_mean(-fake_logit)

    ###TensorBoard visualization###
    self.z_sum = tf.summary.histogram("z", z)
    self.true_sum = tf.summary.histogram("d", true_logit)
    self.fake_sum = tf.summary.histogram("d_", fake_logit)
    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
    self.imaginary_sum = video_summary("imaginary", self.imaginary,self.output_frames+1)

    ###Variable preparing###
    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.d_clamp_op = [tf.assign(var, tf.clip_by_value(var, self.clamp_lower, self.clamp_upper)) for var in self.d_vars]
    
    self.saver = tf.train.Saver()
    self.trainloader=dataloader('/ssd/10.10.20.21/share/guojiaming/UCF-101/Surfing','trainsurfing.txt','train',self.batch_size)
    self.trainloader.start()
    self.testloader=dataloader('/ssd/10.10.20.21/share/guojiaming/UCF-101/Surfing','testsurfing.txt','test',self.batch_size)
    self.testloader.start()
  def train(self, config):
    ################
    # optimization #
    ################
    d_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
               .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.RMSPropOptimizer(config.learning_rate) \
               .minimize(self.g_loss, var_list=self.g_vars)
    ##################
    # initialization #
    ##################
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    ########
    # log  #
    ########
    self.g_sum = tf.summary.merge([self.z_sum, self.fake_sum, self.imaginary_sum, self.g_loss_sum])
    self.d_sum = tf.summary.merge([self.z_sum, self.true_sum, self.d_loss_sum])
    self.writer = tf.summary.FileWriter(self.sampledir+"/logs", self.sess.graph)

    ##################
    # validation set #
    ##################
    sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
    sample_multi_z = [np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim)) for i in range(5)]

    ###################
    # load checkpoint #
    ###################
    if self.load_model:
        could_load, checkpoint_counter = self.load(config.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    counter = 1
    start_time = time.time()
    grid_size = math.sqrt(self.batch_size)
    grid_size=int(grid_size)
    tf.get_default_graph().finalize()
    ###############
    # Start epoch #
    ###############
    
    if 1:
      while counter<self.max_iter:
          
          batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

          if counter < 10 or counter%500 == 0:
            Diter = 10
          else:
            Diter = config.Diter

          ####################
          # Update Critic network #
          ####################
          print("====Update Critic====") 
          for j in range(Diter):
            traindata=self.trainloader.getdata()
            if counter % 100 ==99 and j == 0:
             
              _, summary_str,_ = self.sess.run([d_optim, self.d_sum,self.d_clamp_op], 
                                             feed_dict={self.z: batch_z,self.images:traindata})
              self.writer.add_summary(summary_str, counter) 
            else:
              _,_= self.sess.run([d_optim,self.d_clamp_op], feed_dict={self.z: batch_z,self.images:traindata})

          ####################
          # Update G network #
          ####################
          print("====Update Generator====")
          traindata=self.trainloader.getdata()
          if counter % 100 ==99:
            _, summary_str, errD, errG = self.sess.run([g_optim, self.g_sum, self.d_loss,self.g_loss],
                                              feed_dict={ self.z: batch_z,self.images:traindata})
            self.writer.add_summary(summary_str, counter)
          else:
             _, errD, errG= self.sess.run([g_optim, self.d_loss,self.g_loss],
                                              feed_dict={ self.z: batch_z,self.images:traindata})

          ###########
          # Monitor #
          ###########
          counter += 1
          print(" Counter: [%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
    	           % ( counter, time.time() - start_time, errD, errG))

          ##############
          # validation #
          ##############
          testdata=self.testloader.getdata()
          if np.mod(counter, 5*config.validation) == 1:
            print("~!~!~!~!Multiple Sampling Validation~!~!~!~")
            for i in range(len(sample_multi_z)):
              samples = self.sess.run(self.samplers,feed_dict={self.z: sample_multi_z[i],self.val_images:testdata})
              for times in xrange(self.output_frames+1):
                i_sample = samples[:,times,:,:,:]
                save_images(i_sample, [grid_size,grid_size],
                            self.sampledir+'/train_{:02d}_{:02d}_{:02d}.png'.format(counter, i,times))

          elif np.mod(counter, config.validation) == 2:
            print("~!~!~!~!Single Sampling Validation~!~!~!~")
            samples = self.sess.run(self.samplers, feed_dict={self.z: sample_z,self.val_images:traindata})
            save_gif(samples*255,self.output_frames+1,[grid_size,grid_size],self.sampledir+'/train_{:02d}.gif'.format(counter))
            save_gif(traindata*255,self.output_frames+1,[grid_size,grid_size],self.sampledir+'/real.gif'.format(counter))

          ##############
          # save model #
          ##############
          if np.mod(counter, config.save_times) == 2:
    	      self.save(config.checkpoint_dir, counter)
    '''
    except:
        self.trainloader.close()
        self.testloader.close()
        self.sess.close()
    self.trainloader.close()
    self.testloader.close()
    self.sess.close()
    '''
  def VideoCritic(self, video, reuse=False):
    with tf.variable_scope("VideoCritic") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv3d(video, 64, k_d=4, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv3d(h0, 64*2, k_d=4, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv3d(h1, 64*4, k_d=4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv3d(h2, 64*8, name='d_h3_conv')))
      logits = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

      return logits



  def generator1(self, z, stimage, stimage_64, net_data=None,reuse=False):
    with tf.variable_scope("generator",reuse=reuse) as scope:
      if self.is_flatten:
        flat = tf.reshape(stimage,[self.batch_size,-1])
        f7_fine ,self.f7f_w, self.f7f_b = linear(flat,self.emcode_len,'g_e1_lin',with_w=True)
        e4 = tf.nn.relu(f7_fine)
        emb = tf.concat([e4, z],1)
      elif self.is_conv:
        e0 = lrelu(conv2d(stimage_64, 128, name='g_e0_conv'))
        e1 = lrelu(self.e_bn1(conv2d(e0, 256, name='g_e1_conv')))
        e2 = lrelu(self.e_bn2(conv2d(e1, 512, name='g_e2_conv')))
        e3 = lrelu(self.e_bn3(conv2d(e2, 1024, name='g_e3_conv')))
        e3 = tf.reshape(e3,[self.batch_size,-1])
        e4_1 = linear(e3,self.emcode_len*2,'g_e4_1_lin')
        e4_1 = tf.nn.relu(self.e_bn4(e4_1))
        e4_2 = linear(e4_1,self.emcode_len,'g_e4_2_lin')
        e4_2 = tf.nn.relu(self.e_bn5(e4_2))
        emb = tf.concat([e4_2, z],1)
            ### transformation genrator fc1
      h0 = linear(emb, emb.get_shape()[1]*2, 'g_h0_lin')
      h0 = tf.nn.relu(self.g_bn0(h0))

      ### transformation genrator fc2
      h1 = linear(h0, emb.get_shape()[1], 'g_h1_lin')
      h1 = tf.nn.relu(self.g_bn1(h1))

      kernel_2d_len = self.trans_par*self.sequence_len*self.output_frames
      kernel_3d_len_1 = self.sequence_len*5*5
      kernel_3d_len_2 = self.sequence_len*self.image_height*self.image_width

      ### transformation genrator fc2
      h2 = linear(h1, kernel_2d_len+kernel_3d_len_1+kernel_3d_len_2, 'g_h2_lin')
      kernel_2d = tf.slice(h2,[0,0],[-1,kernel_2d_len])
      kernel_3d_1 = tf.slice(h2,[0,kernel_2d_len],[-1,kernel_3d_len_1])
      kernel_3d_2 = tf.slice(h2,[0,kernel_3d_len_1+kernel_2d_len],[-1,-1])
      print kernel_2d
      ### transformation applying
      if self.transformation == 'affine_transformation':
        self.transformed = affine_apply(stimage_64, kernel_2d, self)
      elif self.transformation == 'conv_transformation':
        self.transformed = conv2d_apply(stimage_64, kernel_2d, self)
      print self.transdormed
      ### Volumetric merge network ###
      frames_1 = volumetric_apply(self.transformed, stimage_64, kernel_3d_1, kernel_3d_2, self)

      firstframe = tf.expand_dims(stimage_64, axis=1)
      video =  [firstframe,frames_1]
      return tf.concat(video,1)
  def generator(self, z, stimage, stimage_64, net_data=None,reuse=False):
    with tf.variable_scope("generator",reuse=reuse) as scope:
      if self.is_flatten:
        flat = tf.reshape(stimage,[self.batch_size,-1])
        f7_fine ,self.f7f_w, self.f7f_b = linear(flat,self.emcode_len,'g_e1_lin',with_w=True)
        e4 = tf.nn.relu(f7_fine)
        emb = tf.concat([e4, z],1)
      elif self.is_conv:
        e0 = lrelu(conv2d(stimage_64, 128, name='g_e0_conv'))
        e1 = lrelu(self.e_bn1(conv2d(e0, 256, name='g_e1_conv')))
        e2 = lrelu(self.e_bn2(conv2d(e1, 512, name='g_e2_conv')))
        e3 = lrelu(self.e_bn3(conv2d(e2, 1024, name='g_e3_conv')))
        e3 = tf.reshape(e3,[self.batch_size,-1])
        e4_1 = linear(e3,self.emcode_len*2,'g_e4_1_lin')
        e4_1 = tf.nn.relu(self.e_bn4(e4_1))
        e4_2 = linear(e4_1,self.emcode_len,'g_e4_2_lin')
        e4_2 = tf.nn.relu(self.e_bn5(e4_2))
        emb = tf.concat([e4_2, z],1)
            ### transformation genrator fc1
      h0 = linear(emb, emb.get_shape()[1]*2, 'g_h0_lin')
      h0 = tf.nn.relu(self.g_bn0(h0))

      ### transformation genrator fc2
      h1 = linear(h0, emb.get_shape()[1], 'g_h1_lin')
      h1 = tf.nn.relu(self.g_bn1(h1))

      kernel_2d_len = self.trans_par*self.sequence_len*self.output_frames
      kernel_3d_len_1 = self.sequence_len*5*5
      kernel_3d_len_2 = self.sequence_len*self.image_height*self.image_width

      ### transformation genrator fc2
      h2 = linear(h1, kernel_2d_len+kernel_3d_len_1+kernel_3d_len_2, 'g_h2_lin')
      kernel_2d = tf.slice(h2,[0,0],[-1,kernel_2d_len])
      kernel_3d_1 = tf.slice(h2,[0,kernel_2d_len],[-1,kernel_3d_len_1])
      kernel_3d_2 = tf.slice(h2,[0,kernel_3d_len_1+kernel_2d_len],[-1,-1])
      ### transformation applying
      if self.transformation == 'affine_transformation':
        self.transformed = affine_apply(stimage_64, kernel_2d, self)
      elif self.transformation == 'conv_transformation':
        self.transformed = conv2d_apply(stimage_64, kernel_2d, self)
      print self.transformed[1].shape
      ### Volumetric merge network ###
      frames_1 = volumetric_apply(self.transformed, stimage_64, kernel_3d_1, kernel_3d_2, self)

      firstframe = tf.expand_dims(stimage_64, axis=1)
      video =  [firstframe,frames_1]
      return tf.concat(video,1)

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.image_height, self.image_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
