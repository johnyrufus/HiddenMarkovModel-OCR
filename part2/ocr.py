#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)
#

from PIL import Image, ImageDraw, ImageFont
from naivebayes import NaiveBayes
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


def simplified_bayes(train_letters, test_letters):
    nb = NaiveBayes(train_letters)
    return ''.join([nb.predict(letter) for letter in test_letters])


def hmm_ve(train_letters, test_letters):
    pass


def hmm_map(train_letters, test_letters):
    pass


def main():
    train_img_fname = 'courier-train.png'
    test_img_fname = 'test-0-0.png'

    train_letters = load_training_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)

    # Simplified
    print('Simple: {}'.format(simplified_bayes(train_letters, test_letters)))

    # HMM VE
    print('HMM VE: {}'.format(hmm_ve(train_letters, test_letters)))

    # HMM MAP
    print('HMM MAP: {}'.format(hmm_map(train_letters, test_letters)))


if __name__ == '__main__':
    main()






