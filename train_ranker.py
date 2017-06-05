# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 18:05:41 2016

@author: jan
"""

import tensorflow as tf
import numpy as np
import time
import imp
tabulate_available = False
try:
    imp.find_module('tabulate')
    tabulate_available = True
except ImportError:
    pass
if tabulate_available:
    from tabulate import tabulate
import argparse

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
      # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 6])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return tf.split(image, 2, 2) # 3rd dimension two parts

def read_and_decode_aug(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
      # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.image.random_flip_left_right(tf.reshape(image, [227, 227, 6]))
  # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.image.random_brightness(image, 0.01)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return tf.split(image, 2, 2) # 3rd dimension two parts

def inputs(filename, batch_size, num_epochs = None, shuffle = False, aug=False):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    if aug:
        crop, full = read_and_decode_aug(filename_queue)
    else:
        crop, full = read_and_decode(filename_queue)

    if shuffle:
        crops, fulls = tf.train.shuffle_batch( [crop, full], batch_size=batch_size,
                                         num_threads=4, capacity=2000 + 4 * batch_size,
                                         enqueue_many = False, min_after_dequeue=1000)
    else:
        crops, fulls = tf.train.batch([crop, full], batch_size = batch_size,
                                      num_threads = 1, capacity=100 + 3 * batch_size,
                                      allow_smaller_final_batch=False)
        # Ensures a minimum amount of shuffling of examples.
#            min_after_dequeue=1000)

    return tf.concat([crops, fulls], 0)


# Helper Functions

def build_loss_matrix(batch_size):
    loss_matrix = np.zeros(shape=(batch_size, batch_size * 2), dtype=np.float32)
    for k in range(batch_size):
        loss_matrix[k,k] = 1
        loss_matrix[k,k+batch_size] = -1
    return loss_matrix

def svm_loss(feature_vec, loss_matrix):
    W = tf.get_variable("W", shape=[feature_vec.get_shape()[1],1], initializer=tf.uniform_unit_scaling_initializer()) # init_weight([int(feature_vec.get_shape()[1]),1])
    q = tf.matmul(feature_vec,W)
    p = tf.matmul(loss_matrix,q)
    zero = tf.constant(0.0, shape=[1], dtype=tf.float32)
    p_hinge = tf.maximum(zero, 1+p)
    L = tf.reduce_mean(p_hinge)
    return L, p

def ranknet_loss(feature_vec, loss_matrix):
    W = tf.get_variable("W", shape=[feature_vec.get_shape()[1],1],
                                    initializer=tf.uniform_unit_scaling_initializer()) # init_weight([int(feature_vec.get_shape()[1]),1])
    q = tf.matmul(feature_vec,W)
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
# Setup Neural Network


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
            bn5 = tf.contrib.layers.flatten(maxpool5, maxpool5.get_shape()[-1])

    flattened_dim = int(np.prod(bn5.get_shape()[1:]))
    fc6W =  tf.get_variable("fc6w", [flattened_dim, embedding_dim], initializer = tf.uniform_unit_scaling_initializer()) # init_weight((flattened_dim, embedding_dim))
    fc6b = tf.get_variable("fc6b", [embedding_dim], initializer = tf.constant_initializer())  #init_bias([embedding_dim])

    fc6 = tf.nn.relu_layer(bn5, fc6W, fc6b)

    return fc6

def count_tfrecords(path):
    cnt = 0
    for record in tf.python_io.tf_record_iterator(path):
        cnt+=1
    return cnt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", help="Embedding dimension before mapping to one-dimensional score", type=int, default = 1000)
    parser.add_argument("--validation_interval", help="Number of iterations after which validation is run", type=int, default = 500)
    #parser.add_argument("--validation_instances", help="Number of validation instances", type=int, default=4040*14)
    parser.add_argument("--batch_train", help="Batch size for training", type=int, default=100)
    parser.add_argument("--batch_val", help="Batch size for validation", type=int, default=14)
    parser.add_argument("--checkpoint_interval", help="Number of iterations after which a checkpoint file is written", type=int, default=1000)
    parser.add_argument("--total_steps", help="Number of total training iterations", type=int, default=15000)
    parser.add_argument("--initial_lr", help="Initial learning rate", type=float, default=0.01)
    parser.add_argument("--momentum", help="Momentum coefficient", type=float, default=0.9)
    parser.add_argument("--step_size", help="Number of steps after which the learning rate is reduced", type=int, default=10000)
    parser.add_argument("--step_factor", help="Reduction factor for the learning rate", type=float, default=0.2)
    parser.add_argument("--initial_parameters", help="Path to initial parameter file", type=str, default="alexnet.npy")
    parser.add_argument("--ranking_loss", help="Type of ranking loss", type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument("--checkpoint_name", help="Name of the checkpoint files", type=str, default='view_finding_network')
    parser.add_argument("--spp", help="Whether to use spatial pyramid pooling in the last layer or not", type=bool, default=True)
    parser.add_argument("--pooling", help="Which pooling function to use", type=bool, choices=['max', 'avg'], default='max')
    parser.add_argument("--augment", help="Whether to augment training data or not", type=bool, default=True)
    parser.add_argument("--training_db", help="Path to training database", type=str, default='trn.tfrecords')
    parser.add_argument("--validation_db", help="Path to validation database", type=str, default='val.tfrecords')
    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    validation_interval = args.validation_interval
    validation_instances = count_tfrecords(args.validation_db)
    batch_size_trn = args.batch_train
    batch_size_val = args.batch_val
    checkpoint_interval = args.checkpoint_interval
    total_steps = args.total_steps

    initial_lr = args.initial_lr
    momentum_coeff = args.momentum
    step_size = args.step_size
    step_factor = args.step_factor

    parameter_path = args.initial_parameters

    ranking_loss = args.ranking_loss

    experiment_name = args.ranking_loss
    spp = args.spp
    augment_training_data = args.augment

    parameter_table = [["Initial parameters", parameter_path],
                    ["Ranking loss", ranking_loss], ["SPP", spp], ["Pooling", args.pooling],
                    ['Experiment', experiment_name],
                    ['Embedding dim', embedding_dim], ['Batch size', batch_size_trn],
                    ['Initial LR', initial_lr], ['Momentum', momentum_coeff],
                    ['LR Step size', step_size], ['LR Step factor', step_factor],
                    ['Total Steps', total_steps]]

    training_images = inputs(args.training_db, batch_size_trn, None, True, augment_training_data)
    test_images = inputs(args.validation_db, batch_size_val, None, False)
    net_data = np.load(parameter_path).item()
    var_dict=  get_variable_dict(net_data)
    with tf.variable_scope("ranker") as scope:
        feature_vec = build_alexconvnet(training_images, var_dict, embedding_dim, spp, args.pooling)
        L, p = loss(feature_vec, build_loss_matrix(batch_size_trn), ranking_loss)
        scope.reuse_variables()
        val_feature_vec = build_alexconvnet(test_images, var_dict, embedding_dim, spp, args.pooling)
        L_val, p_val = loss(val_feature_vec, build_loss_matrix(batch_size_val), ranking_loss)

    lr = tf.Variable(initial_lr)
    opt = tf.train.AdamOptimizer()
    grads = opt.compute_gradients(L)

    apply_grad_op = opt.apply_gradients(grads)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    current_lr = initial_lr

    validation_history = np.zeros(shape=(total_steps/validation_interval, 3))
    if tabulate_available:
        def generate_validation_history(step, tbl):
            return tabulate(tbl, headers=['Step', 'LR', 'Loss'])

        print tabulate(parameter_table)

    for step in range(total_steps+1):
        if step % step_size == 0 and step > 0:
            current_lr *= step_factor
            print "Learning Rate: {}".format(current_lr)
        if step % checkpoint_interval == 0:
            saver.save(sess, 'snapshots/ranker_{}_{}.ckpt'.format(experiment_name, embedding_dim), global_step=step)
        t0 = time.time()
        _, loss_val = sess.run([apply_grad_op, L])
        t1 = time.time()
        print "Iteration {}: L={:0.4f} dT={:0.3f}".format(step, loss_val, t1-t0)
        if step % validation_interval == 0 and step > 0:
            val_avg = 0.0
            for k in range(validation_instances/batch_size_val):
                val_loss = sess.run([L_val])[0]
                val_avg+=val_loss
            val_avg /= float(validation_instances/batch_size_val)
            validation_history[step / validation_interval - 1] = (step, current_lr, val_avg)
            if tabulate_available:
                print generate_validation_history(step/validation_instances, validation_history)
            else:
                print "\tValidation: L={:0.4f}".format(val_avg)
            np.savez("{}_history.npz".format(experiment_name), validation=validation_history)
    if tabulate_available:
        print tabulate(parameter_table)
    sess.close()
