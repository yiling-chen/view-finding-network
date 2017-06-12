# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import skimage.io as io
import skimage.transform as transform
from os.path import join
import network as nw
import json

global_dtype = tf.float32
global_dtype_np = np.float32

# Helper Functions


def overlap_ratio(x1, y1, w1, h1, x2, y2, w2, h2):
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union = (w1 * h1) + (w2 * h2) - intersection
    return float(intersection) / float(union)


def evaluate_sliding_window(img_filename, crops):
    img = io.imread(img_filename).astype(np.float32)/255
    if img.ndim == 2: # Handle B/W images
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, 2)

    scores = np.zeros((len(crops)))

    # compute ranking scores
    for i in xrange(len(crops)):
        crop = crops[i]
        img_crop = transform.resize(img[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2]], (227, 227))-0.5
        img_crop = np.expand_dims(img_crop, axis=0)
        scores[i] = sess.run([score_func], feed_dict={image_placeholder: img_crop})[0]

    # find the optimal crop
    idx = np.argmax(scores)
    best_window = crops[idx]

    # return the best crop
    return (best_window[0], best_window[1], best_window[2], best_window[3])


def evaluate_FCDB():
    slidling_windows_string = open('./data/sliding_window.json', 'r').read()
    sliding_windows = json.loads(slidling_windows_string)

    cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    accum_boundary_displacement = 0
    accum_overlap_ratio = 0

    for item in sliding_windows:
        print 'processing', item['filename']
        crops = item['crops']
        img_filename = join('images', item['filename'])
        img = io.imread(img_filename)
        height = img.shape[0]
        width = img.shape[1]

        # ground truth
        x = crops[0][0]
        y = crops[0][1]
        w = crops[0][2]
        h = crops[0][3]

        best_x, best_y, best_w, best_h = evaluate_sliding_window(img_filename, crops)
        boundary_displacement = (abs(best_x - x) + abs(best_x + best_w - x - w))/float(width) + (abs(best_y - y) + abs(best_y + best_h - y - h))/float(height)
        accum_boundary_displacement += boundary_displacement
        ratio = overlap_ratio(x, y, w, h, best_x, best_y, best_w, best_h)
        if ratio >= alpha:
            alpha_cnt += 1
        accum_overlap_ratio += ratio
        cnt += 1

    print 'Average overlap ratio: {:.4f}'.format(accum_overlap_ratio / cnt)
    print 'Average boundary displacement: {:.4f}'.format(accum_boundary_displacement / (cnt * 4.0))
    print 'Alpha recall: {:.4f}'.format(100 * float(alpha_cnt) / cnt)


def evaluate_aesthetics_score(images):
    scores = np.zeros(shape=(len(images),))
    for i in range(len(images)):
        img = images[i].astype(np.float32)/255
        img_resize = transform.resize(img, (227, 227))-0.5
        img_resize = np.expand_dims(img_resize, axis=0)
        scores[i] = sess.run([score_func], feed_dict={image_placeholder: img_resize})[0]
    return scores


embedding_dim = 1000
ranking_loss = 'svm' # 'ranknet'
snapshot = './snapshots/ranker_svm_1000.ckpt-15000.meta'
net_data = np.load("./alexnet.npy").item()
image_placeholder = tf.placeholder(dtype=global_dtype, shape=[1,227,227,3])
var_dict = nw.get_variable_dict(net_data)
with tf.variable_scope("ranker") as scope:
    feature_vec = nw.build_alexconvnet(image_placeholder, var_dict, embedding_dim)
    score_func = nw.score(feature_vec)

# load pre-trained model
saver = tf.train.Saver(tf.global_variables())
sess = tf.Session(config=tf.ConfigProto())
sess.run(tf.global_variables_initializer())
saver.restore(sess, snapshot)

print "Snapshot: {}".format(snapshot)
evaluate_FCDB()
