#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:56:17 2017

@author: jason
"""

import tensorflow as tf
import numpy as np


def build_loss_matrix(batch_size):
    loss_matrix = np.zeros(shape=(batch_size, batch_size * 2), dtype=np.float32)
    for k in range(batch_size):
        loss_matrix[k,k] = 1
        loss_matrix[k,k+batch_size] = -1
    return loss_matrix

def score(feature_vec):
    W = tf.get_variable("W", shape=[feature_vec.get_shape()[1],1], initializer=tf.uniform_unit_scaling_initializer()) # init_weight([int(feature_vec.get_shape()[1]),1])
    return tf.matmul(feature_vec,W)

def svm_loss(feature_vec, loss_matrix):
    q = score(feature_vec)
    p = tf.matmul(loss_matrix,q)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    p_hinge = tf.maximum(zero, 1+p)
    L = tf.reduce_mean(p_hinge)
    return L, p

def ranknet_loss(feature_vec, loss_matrix):
    q = score(feature_vec)
    p = tf.matmul(loss_matrix,q)
    L = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(p, tf.zeros_like(p), name='RankNetLoss'))
    return L, p

def loss(feature_vec, loss_matrix, ranking_loss_type):
    if ranking_loss_type == 'svm':
        return svm_loss(feature_vec, loss_matrix)
    elif ranking_loss_type == 'ranknet':
        return ranknet_loss(feature_vec, loss_matrix)
    else:
        print "Error: ranking loss >> {} << is unknown".format(ranking_loss_type)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def get_variable_dict(net_data):
    variables_dict = {
        "c1w": tf.Variable(net_data["conv1"][0]),
        "c1b": tf.Variable(net_data["conv1"][1]),

        "c2w": tf.Variable(net_data["conv2"][0]),
        "c2b": tf.Variable(net_data["conv2"][1]),

        "c3w": tf.Variable(net_data["conv3"][0]),
        "c3b": tf.Variable(net_data["conv3"][1]),

        "c4w": tf.Variable(net_data["conv4"][0]),
        "c4b": tf.Variable(net_data["conv4"][1]),

        "c5w": tf.Variable(net_data["conv5"][0]),
        "c5b": tf.Variable(net_data["conv5"][1])}
    return variables_dict

def build_alexconvnet(images, variable_dict, embedding_dim, SPP = False, pooling = 'max'):
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4

    conv1W = variable_dict["c1w"]
    conv1b = variable_dict["c1b"]
    conv1_in = conv(images, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = variable_dict["c2w"]
    conv2b = variable_dict["c2b"]
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)


    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = variable_dict["c3w"]
    conv3b = variable_dict["c3b"]
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)

    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = variable_dict["c4w"]
    conv4b = variable_dict["c4b"]
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)


    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = variable_dict["c5w"]
    conv5b = variable_dict["c5b"]
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    with tf.variable_scope("conv5"):
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        if pooling == 'max':
            pooling_func = tf.nn.max_pool
        else:
            pooling_func = tf.nn.avg_pool
        if SPP:
            maxpool3 = pooling_func(conv5, ksize=[1, 5, 5, 1], strides=[1, 4, 4, 1], padding=padding)
            maxpool2 = pooling_func(conv5, ksize=[1, 7, 7, 1], strides=[1, 6, 6, 1], padding=padding)
            maxpool1 = pooling_func(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding=padding)
            concat5 = tf.concat([tf.contrib.layers.flatten(maxpool1), tf.contrib.layers.flatten(maxpool2), tf.contrib.layers.flatten(maxpool3)], 1)
            bn5 = concat5
        else:
            maxpool5 = pooling_func(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
            bn5 = tf.contrib.layers.flatten(maxpool5)

    flattened_dim = int(np.prod(bn5.get_shape()[1:]))
    fc6W =  tf.get_variable("fc6w", [flattened_dim, embedding_dim], initializer = tf.uniform_unit_scaling_initializer()) # init_weight((flattened_dim, embedding_dim))
    fc6b = tf.get_variable("fc6b", [embedding_dim], initializer = tf.constant_initializer())  #init_bias([embedding_dim])

    fc6 = tf.nn.relu_layer(bn5, fc6W, fc6b)

    return fc6