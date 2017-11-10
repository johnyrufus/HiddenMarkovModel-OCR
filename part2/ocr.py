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
from itertools import chain

ch_width=14
ch_height=25

train_letters_ch = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
initial = defaultdict(int)
trans_prob = dict()
prior_prob = dict()
emission_prob = dict()


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / ch_width) * ch_width, ch_width):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+ch_width) ]) for y in range(0, ch_height) ], ]
    return result


def load_training_letters(fname):
    letter_images = load_letters(fname)
    return { train_letters_ch[i]: letter_images[i] for i in range(0, len(train_letters_ch) ) }


def simplified_bayes(train_letters, test_letters, prior):
    nb = NaiveBayes(train_letters, prior)
    return ''.join([nb.predict(letter) for letter in test_letters])


def hmm_ve(train_letters, test_letters):
    pass


def hmm_map(train_letters, test_letters):
    pass


def calculate_probabilities(fname):
    train_set = set(train_letters_ch)
    for ch1 in train_letters_ch:
        initial[ch1] = 1.0
        prior_prob[ch1] = 1.0
        trans_prob[ch1] = dict()
        for ch2 in train_letters_ch:
            trans_prob[ch1][ch2] = 1.0

    with open(fname) as f:
        for para in f.readlines():
            lines = para.split('. ')
            for i, line in enumerate(lines):
                line = line.lstrip()
                for j, ch in enumerate(line):
                    if ch not in train_set: continue
                    prior_prob[ch] += 1
                    if j == 0:
                        initial[ch] += 1
                    elif j == len(line) - 1 and i != len(lines)-1:
                        trans_prob[ch]['.'] += 1
                    elif line[j-1] in train_set:
                        trans_prob[line[j - 1]][ch] += 1

    initial_total = sum(initial.values())
    prior_total = sum(prior_prob.values())
    for ch1 in train_letters_ch:
        initial[ch1] = initial[ch1]/initial_total
        prior_prob[ch1] = prior_prob[ch1] / prior_total
        trans_total = sum(trans_prob[ch1].values())
        for ch2 in train_letters_ch:
            trans_prob[ch1][ch2] = trans_prob[ch1][ch2] / trans_total


def calculate_error(train_letters, test_letters, naive_prediction):
    total_error = 1
    total_valid = 1
    for i, ch in enumerate(naive_prediction):
        for j, pixel in enumerate(train_letters[ch]):
            if test_letters[i][j] == '*':
                total_error += (1 if pixel != test_letters[i][j] else 0)
                total_valid += (1 if pixel == test_letters[i][j] else 0)
    error_weight = 0.5 # Otherwise the Observation can get completely ignored, if naive bayes prediction is bad
    error_prob = error_weight * total_error / (total_error + total_valid)
    print(error_prob)
    return error_prob


def calculate_emission_prob(train_letters, test_letters, error_prob):
    for ch in train_letters_ch:
        emission_prob[ch] = [0.0] * len(test_letters)
        for i in range(len(test_letters)):
            emission_prob[ch][i] = 1
            for j, pix in enumerate(test_letters[i]):
                emission_prob[ch][i] *= (1 - error_prob) if pix == train_letters[ch][j] else error_prob


def main():
    train_img_fname = 'courier-train.png'
    train_txt_fname = 'DemocracyAndEducation.txt'

    train_letters = load_training_letters(train_img_fname)
    for ch in train_letters_ch:
        train_letters[ch] = list(chain.from_iterable(train_letters[ch]))
    calculate_probabilities(train_txt_fname)

    for i in range(20):
        test_img_fname = 'test-{}-0.png'.format(i)
        test_letters = load_letters(test_img_fname)
        for i, l in enumerate(test_letters):
            test_letters[i] = list(chain.from_iterable(test_letters[i]))


        # Simplified
        simplified_res = simplified_bayes(train_letters, test_letters, prior_prob)
        print('Simple: {}'.format(simplified_res))

        calculate_emission_prob(train_letters, test_letters, calculate_error(train_letters, test_letters, simplified_res))
        # Simplified Without priors
        simplified_res = simplified_bayes(train_letters, test_letters, None)
        calculate_error(train_letters, test_letters, simplified_res)
        print('Simple: {}'.format(simplified_res))

        # HMM VE
        #print('HMM VE: {}'.format(hmm_ve(train_letters, test_letters)))

        # HMM MAP
        #print('HMM MAP: {}'.format(hmm_map(train_letters, test_letters)))


if __name__ == '__main__':
    main()






