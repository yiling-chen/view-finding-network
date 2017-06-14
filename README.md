# View Finding Network

This repository contains the dataset and scripts used in the following article:

[Yi-Ling Chen](https://yiling-chen.github.io/), Jan Klopp, [Min Sun](http://aliensunmin.github.io/), [Shao-Yi Chien](http://www.ee.ntu.edu.tw/profile?id=101), [Kwan-Liu Ma](www.cs.ucdavis.edu/~ma/), ["Learning to Compose with Professional Photographs on the Web"](https://arxiv.org/abs/1702.00503), ACM Multimedia 2017.

**If this work helps your research, please cite the following article:**

    @inproceedings{chen-acmmm-2017,
      title={Learning to Compose with Professional Photographs on the Web},
      author={Yi-Ling Chen and Jan Klopp and Min Sun and Shao-Yi Chien and Kwan-Liu Ma},
      booktitle={ACM Multimedia 2017},
      year={2017}
    }

## Dependencies

You will need to have `tensorflow`, `skimage`, `tabulate`, `pillow` installed on your system to run the scripts.

## Download the dataset

* Clone the repository to your local disk.
* Under a command line window, run the following command to get the images with cropping annotations:
```bash
$ python download_images.py -w 4
```
The above command will launch 4 worker threads to download the images to a default folder (./images) from Flickr.

## Training

* Run `create_dbs.py` to generate the TFRecords files used by Tensorflow.
* Run `train_ranker.py` to start training.
```bash
$ python train_ranker.py --spp 0
```
The above example starts training with SPP disabled. Note that if you changed the output filenames when running `create_dbs.py`, you will need to provide the new filenames to `train_ranker.py`. Take a look at the script to check out other available parameters or run the following command.
```bash
$ python train_ranker.py -h
```

## Evaluation

We provide the evaluation script to reproduce our evaluation results on [Flickr cropping dataset](https://github.com/yiling-chen/flickr-cropping-dataset).
```bash
$ python vfn_eval.py
```
You will need to get `sliding_window.json` and the test images from the website of [Flickr cropping dataset](https://github.com/yiling-chen/flickr-cropping-dataset) and specify the path of your model in `vfn_eval.py`. You can also try our pre-trained model, which can be downloaded from [here]().

## Questions?
If you have questions/suggestions, feel free to send an email to (yiling dot chen dot ntu at gmail dot com).
