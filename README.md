# View Finding Network

This repository contains the dataset and scripts used in the following article:

[Yi-Ling Chen](https://yiling-chen.github.io/), Jan Klopp, [Min Sun](http://aliensunmin.github.io/), [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101), [Kwan-Liu Ma](http://www.cs.ucdavis.edu/~ma/), ["Learning to Compose with Professional Photographs on the Web"](https://arxiv.org/abs/1702.00503), to appear in ACM Multimedia 2017.

## Dependencies

You will need to have `tensorflow` (version > 1.0), `skimage`, `tabulate`, `pillow` installed on your system to run the scripts.

## Download the dataset

* Clone the repository to your local disk.
* Under a command line window, run the following command to get the training images from Flickr:
```bash
$ python download_images.py -w 4
```
The above command will launch 4 worker threads to download the images to a default folder (./images).

## Training

* Run `create_dbs.py` to generate the TFRecords files used by Tensorflow.
* Run `vfn_train.py` to start training.
```bash
$ python vfn_train.py --spp 0
```
The above example starts training with SPP disabled. Or you may want to enable SPP with either `max` or `avg` options.
```bash
$ python vfn_train.py --pooling max
```
Note that if you changed the output filenames when running `create_dbs.py`, you will need to provide the new filenames to `vfn_train.py`. Take a look at the script to check out other available parameters or run the following command.
```bash
$ python vfn_train.py -h
```

## Evaluation

We provide the evaluation script to reproduce our evaluation results on [Flickr cropping dataset](https://github.com/yiling-chen/flickr-cropping-dataset). For example,
```bash
$ python vfn_eval.py --spp false --snapshot snapshots/model-wo-spp
```
You will need to get `sliding_window.json` and the test images from the [Flickr cropping dataset](https://github.com/yiling-chen/flickr-cropping-dataset) and specify the path of your model when running `vfn_eval.py`. You can also try our pre-trained model, which can be downloaded from [here](https://drive.google.com/drive/folders/0B0sDVRDPL5zBd3ozNlFmZEZpY1k?usp=sharing).

## Questions?
If you have questions/suggestions, feel free to send an email to (yiling dot chen dot ntu at gmail dot com).

**If this work helps your research, please cite the following article:**

    @inproceedings{chen-acmmm-2017,
      title={Learning to Compose with Professional Photographs on the Web},
      author={Yi-Ling Chen and Jan Klopp and Min Sun and Shao-Yi Chien and Kwan-Liu Ma},
      booktitle={ACM Multimedia 2017},
      year={2017}
    }
