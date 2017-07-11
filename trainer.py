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

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)
        self.step_update = tf.assign(self.step, self.step + 1)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')

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
            #self.counter = int(next(
            #    re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
            print(' [*] Successfully loaded {}'.format(ckpt_name))
            could_load = True
            return could_load
        else:
            print(' [!] Failed to find a checkpoint')
            could_load = False
            return could_load


    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x)

        self.z = tf.random_uniform(
                (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.z_ot = tf.placeholder(tf.float32, [None, self.z_num], name="z_ot")
        self.x_ot_reshaped = tf.placeholder(tf.float32, 
                [None, self.vectorized_dim], name="x_ot")
        self.input1_enc = tf.placeholder(tf.float32,
                [self.batch_size, self.z_num], name='input1_enc')
        self.input2_enc = tf.placeholder(tf.float32,
                [self.batch_size, self.z_num], name='input2_enc')
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G, self.G_var = GeneratorCNN(
                self.z, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=False)
        G_ot, self.G_ot_var = GeneratorCNN(
                self.z_ot, self.conv_hidden_num, self.channel,
                self.repeat_num, self.data_format, reuse=True)

        d_out, D_z, self.D_var = DiscriminatorCNN(
                tf.concat([G, x], 0), self.channel, self.z_num, self.repeat_num,
                self.conv_hidden_num, self.data_format)
        AE_G, AE_x = tf.split(d_out, 2)
        D_G, self.D_z = tf.split(D_z, 2)

        self.G = denorm_img(G, self.data_format)
        self.G_ot = denorm_img(G_ot, self.data_format)
        self.AE_G = denorm_img(AE_G, self.data_format)
        self.AE_x = denorm_img(AE_x, self.data_format)

        self.G_ot_reshaped = tf.reshape(self.G_ot, [self.batch_size, -1])

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))
        self.coverage_loss = tf.reduce_mean(tf.norm(
            self.G_ot_reshaped - self.x_ot_reshaped, axis=1))
        self.scaled_coverage_loss = (tf.to_float(self.step)/1000000. * 1./7200. *
                self.coverage_loss)
        self.coverage_loss_enc = tf.reduce_mean(tf.norm(
            self.input1_enc - self.input2_enc))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = self.d_loss_fake + self.scaled_coverage_loss
        #NOTE: Original
        #self.g_loss = self.d_loss_fake

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("loss/scaled_coverage_loss", self.scaled_coverage_loss),
            tf.summary.scalar("loss/coverage_loss", self.coverage_loss),
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

        for step in trange(self.start_step, self.max_step):
            # Compute a nearest neighbor set of G's and X's.
            z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

            _, x_ot_r = self.get_gen_z_and_nearest_neighbors(z_ot, method='greedy')

            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }

            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "scaled_coverage_loss": self.scaled_coverage_loss,
                    "coverage_loss": self.coverage_loss,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                })

            result = self.sess.run(fetch_dict,
                feed_dict={
                    self.z_ot: z_ot,
                    self.x_ot_reshaped: x_ot_r})

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                scaled_coverage_loss = result['scaled_coverage_loss']
                coverage_loss = result['coverage_loss']
                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f} Loss_scvg,cvg: {:.4f},{:.4f} measure: {:.4f}, k_t: {:.4f}". \
                    format(step, self.max_step, d_loss, g_loss,
                        scaled_coverage_loss, coverage_loss, measure, k_t))

            if step % (self.log_step * 10) == 0:
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                #cur_measure = np.mean(measure_history)
                #if cur_measure > prev_measure * 0.99:
                #prev_measure = cur_measure

            self.sess.run(self.step_update)

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)
            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
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
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x


    def autoencode(self, inputs, path, idx=None, x_fake=None):
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
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))


    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})


    def decode(self, z):
        # NOTE: This may be wrong. self.D_z is not an ancestor of self.AE_x.
        return self.sess.run(self.AE_x, {self.D_z: z})


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
            # NOTE: This may be wrong. self.D_z is not an ancestor of self.AE_x.
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

            self.interpolate_G(real1_batch, step, root_path, train_epoch=1000)
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
        num_runs = 100

        print("[*] real-real cvg losses...")
        rr_cvg_losses = []
        for _ in range(num_runs):
            x1 = self.get_image_from_loader()
            x2 = self.get_image_from_loader()
            x1r = np.reshape(x1, [self.batch_size, -1])
            x2r = np.reshape(x2, [self.batch_size, -1])
            distances = cdist(x1r, x2r)
            x2r_ot = self.optimal_transport(distances, x2r, method='greedy') 
            rr_coverage_loss = np.mean(LA.norm(x1r - x2r_ot, axis=1))
            rr_cvg_losses.append(rr_coverage_loss)

        print("[*] gen-real cvg losses...")
        gr_cvg_losses = []
        for _ in range(num_runs):
            z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_ot = self.sess.run(self.G_ot, {self.z_ot: z_ot})
            x = self.get_image_from_loader()
            G_ot_r = np.reshape(G_ot, [self.batch_size, -1])
            x_r = np.reshape(x, [self.batch_size, -1])
            distances = cdist(G_ot_r, x_r)
            x_ot_r = self.optimal_transport(distances, x_r, method='greedy') 

            gr_coverage_loss = self.sess.run(self.coverage_loss,
                feed_dict={
                    self.G_ot_reshaped: G_ot_r,
                    self.x_ot_reshaped: x_ot_r})
            gr_cvg_losses.append(gr_coverage_loss)

        print("[*] encoded real-real cvg losses...")
        enc_rr_cvg_losses = []
        for _ in range(num_runs):
            x1 = self.get_image_from_loader()
            x2 = self.get_image_from_loader()
            x1_enc = self.encode(x1)
            x2_enc = self.encode(x2)
            distances = cdist(x1_enc, x2_enc)
            x2_enc_ot = self.optimal_transport(distances, x2_enc, method='greedy') 
            enc_rr_coverage_loss = np.mean(LA.norm(x1_enc - x2_enc_ot, axis=1))
            enc_rr_cvg_losses.append(enc_rr_coverage_loss)

        print("[*] encoded gen-real cvg losses...")
        enc_gr_cvg_losses = []
        for _ in range(num_runs):
            z_ot = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_ot = self.sess.run(self.G_ot, {self.z_ot: z_ot})
            x = self.get_image_from_loader()
            G_enc = self.encode(G_ot)
            x_enc = self.encode(x)
            distances = cdist(G_enc, x_enc)
            x_enc_ot = self.optimal_transport(distances, x_enc, method='greedy') 

            enc_gr_coverage_loss = self.sess.run(self.coverage_loss_enc,
                feed_dict={
                    self.input1_enc: G_enc,
                    self.input2_enc: x_enc_ot})
            enc_gr_cvg_losses.append(enc_gr_coverage_loss)


        print("Sample size: {}, 100 runs".format(self.batch_size))
        print("real-real cvg (min, med, mean, max, var): {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
            np.min(rr_cvg_losses), np.median(rr_cvg_losses), np.mean(rr_cvg_losses),
            np.max(rr_cvg_losses), np.var(rr_cvg_losses)))
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
        # 1. Compute optimal transport/assignment with Greedy or Munkres.
        # 2. Reorder real samples to optimal transport order, for ordered
        #    comparison with generated data.
        # Returns:
        #   x: Tensor of nearest neighbors to Generated input.
        if method == 'greedy':
            num_rows = distances.shape[0]
            num_cols = distances.shape[1]
            indices = []
            d = distances
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


    def get_gen_z_and_nearest_neighbors(self, z_ot, dist='pixel', method='greedy'):
        # 1. Get Generations (or their encodings) for a fixed set of z's.
        # 2. Get a sample (or their encodings) of real data.
        # 3. Vectorize images if needed, for distance calculation.
        # 4. Calculate pair-wise distance matrix, using Euclidean norm.
        G = self.sess.run(self.G_ot, {self.z_ot: z_ot})
        x = self.get_image_from_loader()
        if dist == 'pixel':
            G = np.reshape(G, [self.batch_size, -1])
            x = np.reshape(x, [self.batch_size, -1])
            distances = cdist(G, x)
            x_ot = self.optimal_transport(distances, x, method)
            return G, x_ot
        elif dist == 'encoded':
            G_enc = self.encode(G_ot) 
            x_enc = self.encode(x) 
            distances = cdist(G_enc, x_enc)
            x_enc_ot = self.optimal_transport(distances, x_enc, method)
            return G_enc, x_enc_ot


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


