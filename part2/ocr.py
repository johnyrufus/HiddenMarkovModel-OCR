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
from collections import defaultdict

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

train_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result


def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { train_letters[i]: letter_images[i] for i in range(0, len(train_letters) ) }


def simplified_bayes(train_letters, test_letters):
    nb = NaiveBayes(train_letters)
    return ''.join([nb.predict(letter) for letter in test_letters])


def hmm_ve(train_letters, test_letters):
    pass


def hmm_map(train_letters, test_letters):
    pass


def calculate_initial_transition_probabilities(fname):
    train_set = set(train_letters)
    initial = defaultdict(int)
    trans = {}
    for ch1 in train_letters:
        initial[ch1] = 1.0
        trans[ch1] = {}
        for ch2 in train_letters:
            trans[ch1][ch2] = 1.0
    print(trans)
    print(initial)
    with open(fname) as f:
        for para in f.readlines():
            lines = para.split('. ')
            for i, line in enumerate(lines):
                line = line.lstrip()
                for j, ch in enumerate(line):
                    if ch not in train_set: continue
                    if j == 0:
                        initial[ch] += 1
                    elif j == len(line) - 1 and i != len(lines)-1:
                        trans[ch]['.'] += 1
                    elif line[j-1] in train_set:
                        trans[line[j-1]][ch] += 1
    print(trans)
    print(initial)
    initial_total = sum(initial.values())
    for ch1 in train_letters:
        initial[ch1] = initial[ch1]/initial_total
        trans_total = sum(trans[ch1].values())
        for ch2 in train_letters:
            trans[ch1][ch2] = trans[ch1][ch2]/trans_total
    print(trans)
    print(initial)

    print(sum(initial.values()))
    for ch1 in train_letters:
        print(sum(trans[ch1].values()))










def main():
    train_img_fname = 'courier-train.png'
    test_img_fname = 'test-0-0.png'
    train_txt_fname = 'DemocracyAndEducation.txt'

    train_letters = load_training_letters(train_img_fname)
    test_letters = load_letters(test_img_fname)

    calculate_initial_transition_probabilities(train_txt_fname)

    # Simplified
    print('Simple: {}'.format(simplified_bayes(train_letters, test_letters)))

    # HMM VE
    print('HMM VE: {}'.format(hmm_ve(train_letters, test_letters)))

    # HMM MAP
    print('HMM MAP: {}'.format(hmm_map(train_letters, test_letters)))


if __name__ == '__main__':
    main()






