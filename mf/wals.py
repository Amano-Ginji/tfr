#/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Wals:
    """
    Objective:
        min_{x,y} \sum_{u,i}{c_{u,i}*(p_{u,i}-x_u*y_i)^2+\lambda*(\sum_{u}{|x_u|^2}+\sum_{i}{|y_i|^2})}
    where,
        p_{u,i} = 1 if r_{u,i}>0 else 0
        c_{u,i} = 1+alpha*r_{u,i}
    Note:
        c_{u,i} = r_{u,i} is also ok with gradient descend optimizator
    """

    def __init__(self, user_num, item_num, num_workers, dim=30, alpha=1.0, learning_rate=0.001, l2=1.0, has_bias=False,
            global_step=None, sync_replica=None):
        self.num_workers = num_workers
        self.dim = dim
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.l2 = l2
        self.has_bias = has_bias
        self.global_step = global_step
        self.sync_replicas = sync_replicas
        with tf.device('/job:ps/task:0'):
            self.weight = {
                "user": tf.Variable(tf.truncated_normal([user_num, dim], stddev=0.1), name='user_embedding')
                , "item": tf.Variable(tf.truncated_normal([user_num, dim], stddev=0.1), name='item_embedding')
            }
            if has_bias:
                self.bias = {
                    "global": tf.Variable(tf.zeros([1]), name="global_bias")
                    , "user": tf.Variable(tf.zeros([user_item]), name="user_bias")
                    , "item": tf.Variable(tf.zeros([item_item]), name="item_bias")
                }

    def inference(self, user_batch, item_batch, rate_batch):
        user_embed = tf.nn.embedding_lookup(self.weight["user"], user_batch)
        item_embed = tf.nn.embedding_lookup(self.weight["item"], item_batch)
        infer = tf.reduce_sum(tf.multiply(user_embed, item_embed), 1)
        if has_bias:
            user_bias = tf.nn.embedding_lookup(self.bias["user"], user_batch)
            item_bias = tf.nn.embedding_lookup(self.bias["item"], item_batch)
            infer = tf.add(infer, self.bias["global"])
            infer = tf.add(infer, user_bias)
            infer = tf.add(infer, item_bias, name="wals_infer")
        reg = tf.add(tf.nn.l2_loss(user_embed), tf.nn.l2_loss(item_embed), name="wals_reg")
        return infer, reg

    def train_batch(self, sess, user_batch, item_batch, rate_batch):
        infer, reg = self.inference(user_batch, item_batch)
        c_ui = tf.add(1.0, tf.multiply(self.alpha, rate_batch))
        p_ui = self.binary_rate(sess, rate_batch)
        loss = tf.reduce_mean(tf.multiply(c_ui, tf.pow(p_ui-infer, 2)), 0)
        cost = loss+self.l2*reg

        opt = tf.train.AdamOptimizer(self.learning_rate)
        if self.sync_replicas is not None:
            opt = tf.train.SyncReplicasOptimizerV2(opt
                    , replicas_to_aggregate=self.num_workers*self.sync_replicas
                    , total_num_replicas=self.num_workers
                    , name="wals_sync_replicas")
        return (train_op = opt.minimize(cost, global_step=self.global_step))

    def binary_rate(self, sess, rate_batch):
        with sess.as_default():
            np_rate = np.array(rate_batch.eval())
            np_rate[np_rate>0] = 1
            return tf.convert_to_tensor(np_rate)

