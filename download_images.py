#!/usr/bin/env python
import os
import urllib
import argparse
import multiprocessing
import cPickle as pkl
from PIL import Image

image_folder = './images/'

def fetch_image(url):
    filename = os.path.split(url)[-1]
    full_path = os.path.join(image_folder, filename)
    if os.path.exists(full_path):
        return

    print '\tDownloading', filename
    file, mime = urllib.urlretrieve(url)
    photo = Image.open(file)
    photo.save(full_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the images in the dataset into a specified folder.')
    parser.add_argument(
        '-w', '--workers', type=int, default=-1,
        help="num workers used to download images. -x uses (all - x) cores [-1 default]."
    )
    parser.add_argument('-dir', type=str, default='./images/',
        help='the path to save the images, default="./images/"'
    )
    args = parser.parse_args()
    image_folder = args.dir
    num_workers = args.workers

    if num_workers < 0:
        num_workers = multiprocessing.cpu_count() + num_workers

    if not os.path.exists(image_folder):
        print 'Creating folder to download images...[{}]'.format(image_folder)
        os.makedirs(image_folder)

    db = pkl.load(open("dataset.pkl", "rb"))
    URLs = [db[i]['url'] for i in xrange(0, len(db), 14)]

    print('Downloading {} images with {} workers...'.format(len(URLs), num_workers))
    pool = multiprocessing.Pool(processes=num_workers)
    pool.map(fetch_image, URLs)
