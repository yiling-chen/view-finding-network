# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 18:28:02 2016

@author: jan
"""
import numpy as np
import skimage.transform as transform
import skimage.io as io
import tensorflow as tf
import cPickle as pkl
import os
import re
import argparse

cnn_input = (227,227)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_database(tfr_file, image_folder, mtdb, offset, n, size, crops, n_crops):
    expr = re.compile(".*/([0-9_a-f]*\.jpg)")

    print "Writing {} crops of {} images to {}".format(len(crops), n, tfr_file)
    with tf.python_io.TFRecordWriter(tfr_file) as writer:
        k = 0
        while k < n:
            idx = (k+offset)*n_crops
            info = mtdb[idx]
            match = expr.match(info['url'])
            img_path = os.path.join(image_folder, match.group(1))
            # skip images of small size, which is very likely to be an image already deleted by user
            img_info = os.stat(img_path)
            if img_info.st_size < 9999:
                k += 1
                continue
            img = io.imread(img_path).astype(np.float32)/255.
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3,2)
            img_full = transform.resize(img, size)
            for l in crops:
                try:
                    idx_crop = idx+l
                    info = mtdb[idx_crop]
                    crop = info['crop']
                    img_crop = transform.resize(img[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2]], size)
                    img_comb = (np.append(img_crop, img_full, axis = 2)*255.).astype(np.uint8)
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(size[0]),
                        'width': _int64_feature(size[1]),
                        'depth': _int64_feature(6),
                        'image_raw': _bytes_feature(img_comb.tostring()),
                        'img_file': _bytes_feature(match.group(1)),
                        'crop': _bytes_feature(np.array(crop).tostring()),
                        'crop_type': _bytes_feature(info['crop_type']),
                        'crop_scale': _float_feature(info['crop_scale'])}))
                    writer.write(example.SerializeToString())
                except:
                    print "Error processing image crop {} of image {}".format(l, match.group(1))
                    pass
            if (k+1) % 100 == 0:
                print "Wrote {} examples".format(k+1)
            k += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_db", help="Path to training database", type=str, default="trn.tfrecords")
    parser.add_argument("--validation_db", help="Path to validation database", type=str, default="val.tfrecords")
    parser.add_argument("--image_folder", help="Folder containing training & validation images as downloaded from Flickr", type=str, default="images/")
    parser.add_argument("--n_trn", help="Number of training images", type=int, default=17000)
    parser.add_argument("--n_val", help="Number of validation images", type=int, default=4040)
    parser.add_argument("--crop_data", help="Path to crop database", type=str, default="dataset.pkl")
    parser.add_argument("--n_crops", help="Number of crops per image", type=int, default=14)
    args = parser.parse_args()

    with open(args.crop_data, 'r') as f:
        crop_db = pkl.load(f)

    n_images = int(len(crop_db)/args.n_crops)

    if (n_images < args.n_trn + args.n_val) :
        print "Error: {} images available, {} required for train/validation".format(n_images, args.n_trn+args.n_val)
        exit()
    offset_val = create_database(args.training_db, args.image_folder, crop_db, 0,
            args.n_trn, cnn_input, xrange(args.n_crops), args.n_crops)
    val_images = create_database(args.validation_db, args.image_folder, crop_db, offset_val,
            args.n_val, cnn_input, xrange(args.n_crops), args.n_crops)
