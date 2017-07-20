from __future__ import print_function

import numpy as np
import os
import pdb
import re
import scipy.misc
import StringIO
import sys
import time
from collections import deque
from glob import glob
from itertools import chain
from munkres import Munkres
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import trange

from data_loader import get_loader
from models import *
from utils import save_image


class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.split = config.split
        self.coverage_diagnostics = config.coverage_diagnostics
        self.coverage_space = config.coverage_space
        self.coverage_norm_order = config.coverage_norm_order
        self.train_ae = config.train_ae
        self.train_gan = config.train_gan

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)
        self.step_update = tf.assign(self.step, self.step + 1)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5,
            config.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5,
            config.lr_lower_boundary), name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.log_dir = config.log_dir
        self.load_path = config.load_path

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
                get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2
        self.vectorized_dim = (self.input_scale_size * self.input_scale_size *
                self.channel)

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                is_chief=True,
                                saver=self.saver,
                                summary_op=None,
                                summary_writer=self.summary_writer,
                                save_model_secs=300,
                                global_step=self.step,
                                ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        # Train or test, depending on config.
        if self.is_train and self.load_path is not None:
            could_load = self.load_checkpoints()
            if could_load:
                self.max_step = self.step + (self.max_step - self.start_step)
                self.start_step = self.step
        elif not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False
            self.build_test_model()
            could_load = self.load_checkpoints()

        # Run diagnostic of coverage loss on real data.
        if self.coverage_diagnostics:
            self.coverage_loss_on_real_data()


    def load_checkpoints(self):
        # TODO: Sort out why checkpoint step is being doubled.
        # E.g. 1000_G.png is assocaited with checkpoint 2000.
        print(' [*] Reading checkpoints')
        ckpt = tf.train.get_checkpoint_state(self.load_path)

        if ckpt and ckpt.model_checkpoint_path:

            # Check if user wants to continue.
            user_input = raw_input(
                'Found checkpoint {}. Proceed? (y/n) '.format(
                    ckpt.model_checkpoint_path))
            if user_input != 'y':
                raise ValueError(
                    ' [!] Cancelled. To start fresh, rm checkpoint files.')

            # Rewrite any necessary variables, based on loaded ckpt.
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.load_path,
                ckpt_name))
            self.step = int(
                re.search('(\d+)(?!.*\d)', ckpt_name).group(0)) + 1
            print(' [*] Successfully loaded {}'.format(ckpt_name))
            could_load = True
            return could_load
        else:
            print(' [!] Failed to find a checkpoint')
            could_load = False
            return could_load


    def build_model(self):
        # Inputs.
        self.x = self.data_loader
        x = norm_img(self.x)

        self.z = tf.random_uniform(
                (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.jitter = tf.truncated_normal(
                (self.batch_size, self.z_num), mean=0.0, stddev=0.1)
        self.z_jitter = tf.clip_by_value(self.z + self.jitter, -1.0, 1.0)
        self.z_jitter_ = tf.clip_by_value(self.z - self.jitter, -1.0, 1.0)
        self.z_ot = tf.placeholder(tf.float32, [None, self.z_num], name="z_ot")
        self.x_ot_reshaped = tf.placeholder(tf.float32, 
                [None, self.vectorized_dim], name="x_ot")
        self.input1_enc = tf.placeholder(tf.float32,
                [self.batch_size, self.z_num], name='input1_enc')
        self.input2_enc = tf.placeholder(tf.float32,
                [self.batch_size, self.z_num], name='input2_enc')
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        # Network outputs.
        G_out, self.G_var = GeneratorCNN(
                tf.concat([self.z, self.z_jitter, self.z_jitter_], 0),
                self.conv_hidden_num, self.channel, self.repeat_num,
                self.data_format, reuse=False)
        G, G_jit, G_jit_ = tf.split(G_out, 3)
        G_ot, _ = GeneratorCNN(
                self.z_ot, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=True)

        d_out, d_enc, self.D_var = DiscriminatorCNN(
                tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format, reuse=False)
        AE_G, AE_x = tf.split(d_out, 2)
        _, self.x_enc = tf.split(d_enc, 2)
        _, self.G_enc, _ = DiscriminatorCNN(
                tf.concat([G_ot], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format, reuse=True)

        # Rescaling/reshaping outputs.
        self.G = denorm_img(G, self.data_format)
        self.G_ot = denorm_img(G_ot, self.data_format)
        self.G_jit = denorm_img(G_jit, self.data_format)
        self.G_jit_ = denorm_img(G_jit_, self.data_format)
        self.AE_G = denorm_img(AE_G, self.data_format)
        self.AE_x = denorm_img(AE_x, self.data_format)
        self.G_ot_reshaped = tf.reshape(self.G_ot, [self.batch_size, -1])

        # Set up optimizers, and their respective learning rates.
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))
        d_optimizer = optimizer(self.d_lr)
        g_optimizer = optimizer(self.g_lr)
        cvg_optimizer = optimizer(self.g_lr)
        smo_optimizer = optimizer(0.00001)
        vol_optimizer = optimizer(0.00001)
        var_optimizer = optimizer(0.000005)

        # Define losses.
        self.d_loss_real = tf.reduce_mean(tf.norm(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.norm(AE_G - G))

        self.smoothness_loss = tf.reduce_mean(tf.norm(G - G_jit))
        self.volume_loss = tf.reduce_mean(tf.norm(G_jit - 2 * G + G_jit_))

        _, variance = tf.nn.moments(self.G_enc, axes=[1])
        self.variance_loss = -1 * tf.reduce_mean(variance)

        self.coverage_loss_pixel = tf.reduce_mean(tf.norm(
            self.G_ot_reshaped - self.x_ot_reshaped,
            ord=self.coverage_norm_order, axis=1))
        self.coverage_loss_enc_manual = tf.reduce_mean(tf.norm(self.input1_enc -
            self.input2_enc, ord=self.coverage_norm_order))
        self.coverage_loss_enc = tf.reduce_mean(tf.norm(self.G_enc - self.x_enc,
            ord=self.coverage_norm_order))
        if self.coverage_space == 'pixel':
            self.coverage_loss = self.coverage_loss_pixel
        elif self.coverage_space == 'encoding':
            self.coverage_loss = self.coverage_loss_enc
        else:
            raise ValueError('self.coverage_space must be \'pixel\' or \'encoding\'')

        self.ae_loss = self.d_loss_real
        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = self.d_loss_fake
        self.cvg_loss = self.coverage_loss
        self.smo_loss = self.smoothness_loss
        self.vol_loss = self.volume_loss
        self.var_loss = self.variance_loss

        # Define optimization nodes.
        self.ae_optim = d_optimizer.minimize(self.ae_loss, var_list=self.D_var,
                global_step=self.step)
        self.d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var,
                global_step=self.step)
        self.g_optim = g_optimizer.minimize(self.g_loss, var_list=self.G_var)
        self.cvg_optim = cvg_optimizer.minimize(self.cvg_loss, var_list=self.G_var)
        self.smo_optim = smo_optimizer.minimize(self.smo_loss, var_list=self.G_var)
        self.vol_optim = vol_optimizer.minimize(self.vol_loss, var_list=self.G_var)
        self.var_optim = var_optimizer.minimize(self.var_loss, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([self.d_optim, self.g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_x", self.AE_x),
            tf.summary.image("x", denorm_img(x, self.data_format)),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/coverage_loss", self.coverage_loss),
            tf.summary.scalar("loss/smoothness_loss", self.smoothness_loss),
            tf.summary.scalar("loss/volume_loss", self.volume_loss),
            tf.summary.scalar("loss/variance_loss", self.variance_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])


    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))
        prev_measure = 1
        measure_history = deque([0]*self.lr_update_step, self.lr_update_step)
        print('Coverage norm order: {}'.format(self.coverage_norm_order))

        # Pretrain autoencoder.
        if self.train_ae:
            ae_steps = 5000
            z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            fetch_dict_ae_logs = {
                'summary': self.summary_op,
                'd_loss_real': self.d_loss_real,
                'coverage_loss': self.coverage_loss,
                'smoothness_loss': self.smoothness_loss,
                'volume_loss': self.volume_loss,
                'variance_loss': self.variance_loss}
            print('Pretraining autoencoder with {} steps.'.format(ae_steps))
            for ae_step in xrange(ae_steps):
                self.sess.run(self.ae_optim)
                if ae_step % self.log_step == 0:
                    result = self.sess.run(fetch_dict_ae_logs,
                        {self.z_ot: z_ot})
                    self.summary_writer.add_summary(result['summary'], ae_step)
                    self.summary_writer.flush()

                    d_loss_real = result['d_loss_real']
                    coverage_loss = result['coverage_loss']
                    smoothness_loss = result['smoothness_loss']
                    volume_loss = result['volume_loss']
                    variance_loss = result['variance_loss']

                    print('"[{}/{}] d_loss_real: {:.4f} coverage: {:.4f} smooth: {:.4f} volume: {:.4f} variance: {:.4f}'. \
                        format(ae_step, ae_steps, d_loss_real, coverage_loss,
                            smoothness_loss, volume_loss, variance_loss))

        # Train full GAN.
        if self.train_gan:
            print('Training GAN with {} steps.'.format(
                self.max_step - self.start_step))
            for step in trange(self.start_step, self.max_step):
                # Compute a nearest neighbor set of G's and X's.
                z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
                x = self.get_image_from_loader()
                #x_ot = self.reorder_x(z_ot, x, dist=self.coverage_space,
                #    method='greedy')
                x_ot = x
                x_ot_reshaped = np.reshape(x_ot, [self.batch_size, -1])

                fetch_dict = {
                    'k_update': self.k_update,
                    #'cvg_optim': self.cvg_optim,
                    #'smo_optim': self.smo_optim,
                    #'vol_optim': self.vol_optim,
                    'var_optim': self.var_optim,
                }

                if step % self.log_step == 0:
                    fetch_dict.update({
                        'summary': self.summary_op,
                        'coverage_loss': self.coverage_loss,
                        'smoothness_loss': self.smoothness_loss,
                        'volume_loss': self.volume_loss,
                        'variance_loss': self.variance_loss,
                        'd_loss_real': self.d_loss_real,
                        'd_loss_fake': self.d_loss_fake,
                        'k_t': self.k_t,
                    })

                if self.coverage_space == 'encoding':
                    result = self.sess.run(fetch_dict,
                        feed_dict={
                            self.z_ot: z_ot,
                            self.x: to_nchw_numpy(x_ot)})
                elif self.coverage_space == 'pixel':
                    result = self.sess.run(fetch_dict,
                        feed_dict={
                            self.z_ot: z_ot,
                            self.x_ot_reshaped: x_ot_reshaped})
                else:
                    raise ValueError('self.coverage_space must be \'encoding\' or \'pixel\'')


                if step % self.log_step == 0:
                    self.summary_writer.add_summary(result['summary'], step)
                    self.summary_writer.flush()

                    coverage_loss = result['coverage_loss']
                    smoothness_loss = result['smoothness_loss']
                    volume_loss = result['volume_loss']
                    variance_loss = result['variance_loss']
                    d_loss_real = result['d_loss_real']
                    d_loss_fake = result['d_loss_fake']
                    k_t = result['k_t']

                    print('[{}/{}] d_loss_real/fake: {:.4f}/{:.4f} Loss_cvg: {:.4f} Smooth: {:.4f}, Volume: {:.4f}, Variance: {:.4f}, k_t: {:.4f}'. \
                        format(step, self.max_step, d_loss_real, d_loss_fake,
                            coverage_loss, smoothness_loss, volume_loss,
                            variance_loss, k_t))

                if step % (self.log_step * 10) == 0:
                    x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                    self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

                    # Save diagnostics.
                    self.diagnostic(step)

                if step % self.lr_update_step == self.lr_update_step - 1:
                    self.sess.run([self.g_lr_update, self.d_lr_update])


    def diagnostic(self, step):
        g_optim_num = 1

        # Save current checkpoint.
        #ckpt = tf.train.get_checkpoint_state(self.model_dir)
        #ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        # Do stuff.
        z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
        x = self.get_image_from_loader()
        G = self.sess.run(self.G_ot, {self.z_ot: z})
        G_enc = self.encode(G)
        x_enc = self.encode(x)
        AE_G = self.autoencode(G, self.model_dir, save=False)
        AE_x = self.autoencode(x, self.model_dir, save=False)
        distances = cdist(G_enc, x_enc)
        x_ot = self.optimal_transport(distances, x, method='greedy') 

        for _ in range(g_optim_num):
            self.sess.run(self.g_optim, feed_dict={
                self.z_ot: z,
                self.x: to_nchw_numpy(x_ot)})

        G_ = self.sess.run(self.G_ot, {self.z_ot: z})
        G_enc_ = self.encode(G)
        x_enc_ = self.encode(x)
        AE_G_ = self.autoencode(G, self.model_dir, save=False)
        AE_x_ = self.autoencode(x, self.model_dir, save=False)
        
        # Restore original.
        #self.saver.restore(self.sess, os.path.join(self.load_path,
        #    ckpt_name))

        # Save summary image.
        big_img = np.stack([x, G, G_], 0).transpose([0,2,1,3,4]).reshape(
            [self.input_scale_size * 3,
             self.input_scale_size * self.batch_size,
             3])
        big_img = np.rint(big_img).astype(np.uint8)
        im = Image.fromarray(big_img)
        save_path = os.path.join(self.model_dir, 
                'big_img_{}.png'.format(step))
        im.save(save_path)
        print('Saved {}'.format(save_path))


    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)
            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num],
                tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _ = GeneratorCNN(
            self.z_r, self.conv_hidden_num, self.channel,
            self.repeat_num, self.data_format, reuse=True)

        # Optimize z so that G(z) matches X. Give z_r as preimage of x.
        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))


    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, 'G_{}.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x


    def autoencode(self, inputs, path, idx=None, x_fake=None, save=True):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            if save:
                save_image(x, x_path)
                print("[*] Samples saved: {}".format(x_path))
        return x


    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.x_enc, {self.x: inputs})


    def decode(self, z):
        # NOTE: This may be wrong. self.x_enc is not an ancestor of self.AE_x.
        return self.sess.run(self.AE_x, {self.x_enc: z})


    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size/2)

        # Optimize z so that G(z) matches X. Give z_r as preimage of x.
        # When train_epoch=0, z = self.z_r = self.z, as defined in build_model().
        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            print('Optimizing z as preimage of sampled real batch.')
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)
        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])

        #for idx, img in enumerate(generated):
        #    save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)


    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            # NOTE: This may be wrong. self.x_enc is not an ancestor of self.AE_x.
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)


    def test(self):
        test_output_dir = os.path.join(self.model_dir, 'test')
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        root_path = test_output_dir 

        all_G_z = None
        for step in range(3):
            real1_batch = self.get_image_from_loader()
            real2_batch = self.get_image_from_loader()

            #save_image(real1_batch, os.path.join(root_path,
            #    'test{}_real1.png'.format(step)))
            #save_image(real2_batch, os.path.join(root_path,
            #    'test{}_real2.png'.format(step)))

            #self.autoencode(
            #    real1_batch, root_path, idx="test{}_real1".format(step))
            #self.autoencode(
            #    real2_batch, root_path, idx="test{}_real2".format(step))

            self.interpolate_G(real1_batch, step, root_path, train_epoch=100)
            #self.interpolate_D(real1_batch, real2_batch, step, root_path)

            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, root_path=root_path, 
                idx="test{}_z_fixed".format(step), save=False)

            if all_G_z is None:
                all_G_z = G_z
            else:
                all_G_z = np.concatenate([all_G_z, G_z])

        save_image(all_G_z, os.path.join(root_path, 'all_G_z.png'), nrow=16)


    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


    def coverage_loss_on_real_data(self):
        if not self.load_path:
            raise ValueError(('[!] This test requires a load_path to fetch an '
                'existing encoder. None found.'))

        from numpy import linalg as LA
        num_runs = 20

        print("[*] real-real cvg losses...")
        rr_greedy = []
        rr_munkres = []
        for _ in range(num_runs):
            x1 = self.get_image_from_loader()
            x2 = self.get_image_from_loader()
            x1r = np.reshape(x1, [self.batch_size, -1])
            x2r = np.reshape(x2, [self.batch_size, -1])
            distances = cdist(x1r, x2r)

            x2r_ot_g = self.optimal_transport(distances, x2r, method='greedy') 
            rr_greedy_ = np.mean(LA.norm(x1r - x2r_ot_g, axis=1))
            rr_greedy.append(rr_greedy_)
            
            x2r_ot_m = self.optimal_transport(distances, x2r, method='munkres') 
            rr_munkres_ = np.mean(LA.norm(x1r - x2r_ot_m, axis=1))
            rr_munkres.append(rr_munkres_)

        print("[*] gen-real cvg losses...")
        gr_greedy = []
        gr_munkres = []
        for _ in range(num_runs):
            z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_ot = self.sess.run(self.G_ot, {self.z_ot: z_ot})
            x = self.get_image_from_loader()
            G_ot_r = np.reshape(G_ot, [self.batch_size, -1])
            x_r = np.reshape(x, [self.batch_size, -1])
            distances = cdist(G_ot_r, x_r)

            x_ot_r_g = self.optimal_transport(distances, x_r, method='greedy') 
            gr_greedy_ = self.sess.run(self.coverage_loss_pixel,
                feed_dict={
                    self.G_ot_reshaped: G_ot_r,
                    self.x_ot_reshaped: x_ot_r_g})
            gr_greedy.append(gr_greedy_)

            x_ot_r_m = self.optimal_transport(distances, x_r, method='munkres') 
            gr_munkres_ = self.sess.run(self.coverage_loss_pixel,
                feed_dict={
                    self.G_ot_reshaped: G_ot_r,
                    self.x_ot_reshaped: x_ot_r_m})
            gr_munkres.append(gr_munkres_)

        print("[*] encoded real-real cvg losses...")
        enc_rr_greedy = []
        enc_rr_munkres = []
        for _ in range(num_runs):
            x1 = self.get_image_from_loader()
            x2 = self.get_image_from_loader()
            x1_enc = self.encode(x1)
            x2_enc = self.encode(x2)
            distances = cdist(x1_enc, x2_enc)

            x2_enc_ot_g = self.optimal_transport(distances, x2_enc, method='greedy') 
            enc_rr_greedy_ = np.mean(LA.norm(x1_enc - x2_enc_ot_g, axis=1))
            enc_rr_greedy.append(enc_rr_greedy_)

            x2_enc_ot_m = self.optimal_transport(distances, x2_enc, method='munkres') 
            enc_rr_munkres_ = np.mean(LA.norm(x1_enc - x2_enc_ot_m, axis=1))
            enc_rr_munkres.append(enc_rr_munkres_)

        print("[*] encoded gen-real cvg losses...")
        enc_gr_greedy = []
        enc_gr_munkres = []
        for _ in range(num_runs):
            z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_ot = self.sess.run(self.G_ot, {self.z_ot: z_ot})
            x = self.get_image_from_loader()
            G_enc = self.encode(G_ot)
            x_enc = self.encode(x)
            distances = cdist(G_enc, x_enc)

            x_enc_ot_g = self.optimal_transport(distances, x_enc, method='greedy') 
            enc_gr_greedy_ = self.sess.run(self.coverage_loss_enc_manual,
                feed_dict={
                    self.input1_enc: G_enc,
                    self.input2_enc: x_enc_ot_g})
            enc_gr_greedy.append(enc_gr_greedy_)

            x_enc_ot_m = self.optimal_transport(distances, x_enc, method='munkres') 
            enc_gr_munkres_ = self.sess.run(self.coverage_loss_enc_manual,
                feed_dict={
                    self.input1_enc: G_enc,
                    self.input2_enc: x_enc_ot_m})
            enc_gr_munkres.append(enc_gr_munkres_)



        with open("coverage_loss_diagnostics.txt", "a") as f:
            f.write("\nrr greedy. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(rr_greedy), np.median(rr_greedy), np.mean(rr_greedy),
            np.max(rr_greedy), np.var(rr_greedy)))
            f.write("\nrr munkres. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(rr_munkres), np.median(rr_munkres), np.mean(rr_munkres),
            np.max(rr_munkres), np.var(rr_munkres)))
            f.write("\ngr greedy. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(gr_greedy), np.median(gr_greedy), np.mean(gr_greedy),
            np.max(gr_greedy), np.var(gr_greedy)))
            f.write("\ngr munkres. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(gr_munkres), np.median(gr_munkres), np.mean(gr_munkres),
            np.max(gr_munkres), np.var(gr_munkres)))
            f.write("\nenc rr greedy. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(enc_rr_greedy), np.median(enc_rr_greedy), np.mean(enc_rr_greedy),
            np.max(enc_rr_greedy), np.var(enc_rr_greedy)))
            f.write("\nenc rr munkres. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(enc_rr_munkres), np.median(enc_rr_munkres), np.mean(enc_rr_munkres),
            np.max(enc_rr_munkres), np.var(enc_rr_munkres)))
            f.write("\nenc gr greedy. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(enc_gr_greedy), np.median(enc_gr_greedy), np.mean(enc_gr_greedy),
            np.max(enc_gr_greedy), np.var(enc_gr_greedy)))
            f.write("\nenc gr munkres. (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(enc_gr_munkres), np.median(enc_gr_munkres), np.mean(enc_gr_munkres),
            np.max(enc_gr_munkres), np.var(enc_gr_munkres)))

        sys.exit('ended coverage loss comparison')


        print("Sample size: {}, 20 runs".format(self.batch_size))
        print("real-real cvg (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(rr_greedy), np.median(rr_greedy), np.mean(rr_greedy),
            np.max(rr_greedy), np.var(rr_greedy)))
        print("real-gen cvg (min, med, mean,  / max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(gr_cvg_losses), np.median(gr_cvg_losses), np.mean(gr_cvg_losses),
            np.max(gr_cvg_losses), np.var(gr_cvg_losses)))
        print('\n')
        print("encoded real-real cvg (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(enc_rr_cvg_losses), np.median(enc_rr_cvg_losses), np.mean(enc_rr_cvg_losses),
            np.max(enc_rr_cvg_losses), np.var(enc_rr_cvg_losses)))
        print("encoded gen-real cvg (min, med, mean,  / max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(enc_gr_cvg_losses), np.median(enc_gr_cvg_losses), np.mean(enc_gr_cvg_losses),
            np.max(enc_gr_cvg_losses), np.var(enc_gr_cvg_losses)))
        print("\n")
        sys.exit('ended coverage loss comparison')


    def optimal_transport(self, distances, x, method='greedy'):
        # Orders target data set, x, according to optimal transport algorithm.
        #
        # 1. Compute optimal transport/assignment with Greedy or Munkres.
        # 2. Reorder real samples to optimal transport order, for ordered
        #    comparison with generated data.
        # Note: The generated data G has an ordering implicit in the distances
        #   array. This function returns only the reordered target set, x.
        #
        # Args:
        #   distances: A 2D numpy array with pairwise distances
        #   x: A 2D numpy array of 
        # Returns:
        #   x_ot: Tensor of nearest neighbors to Generated input, i.e. a reordering 
        #      according to optimal transport.
        if method == 'greedy':
            num_rows = distances.shape[0]
            num_cols = distances.shape[1]
            indices = []
            d = np.copy(distances)
            for _ in xrange(num_rows):
                k = np.argmin(d)
                min_row = k/num_cols
                min_col = k%num_rows
                indices.append((min_row, min_col))
                # Set entire row and col of min value to Infinity.
                for i in xrange(num_rows):
                    d[i][min_col] = float('inf')
                for j in xrange(num_cols):
                    d[min_row][j] = float('inf')
            indices = sorted(indices, key=lambda x: x[0])
        else:
            # Munkres() already orders indices by first value.
            indices = Munkres().compute(distances)

        x_ot = np.array([x[j] for (i,j) in indices])
        return x_ot


    def reorder_x(self, z_ot, x, dist='encoding', method='greedy'):
        # 1. Get Generations (or their encodings) for a fixed set of z's.
        # 2. Get a sample (or their encodings) of real data.
        # 3. Vectorize images if needed, for distance calculation.
        # 4. Calculate pair-wise distance matrix, using Euclidean norm.
        # NOTE: Argument x is the full image sample.
        G = self.sess.run(self.G_ot, {self.z_ot: z_ot})
        if dist == 'pixel':
            G_r = np.reshape(G, [self.batch_size, -1])
            x_r = np.reshape(x, [self.batch_size, -1])
            distances = cdist(G_r, x_r)
            x_ot = self.optimal_transport(distances, x, method)
            return x_ot
        elif dist == 'encoding':
            G_enc = self.encode(G) 
            x_enc = self.encode(x) 
            distances = cdist(G_enc, x_enc)
            x_ot = self.optimal_transport(distances, x, method)
            return x_ot


def next(loader):
    return loader.next()[0].data.numpy()


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image


def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image


def norm_img(image, data_format=None):
    image = image/127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1)*127.5, data_format), 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


