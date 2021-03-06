#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Johny Rufus
# (based on skeleton code by D. Crandall, Oct 2017)
#

'''
Some of the design choices made:

1. Comparing two images by comparing the pixels proved to be a pretty tricky issue,
as comparing the pixels by just taking into account the matching pixels(using both
black and white) would lead a lot of characters to be predicted as ' '(Space).
Just taking into account the pixels that are ON (black), solved the above problem,
but this will end up in not predicting a ' '  correctly, even under slight noise.
So the final idea that worked was, using a combination of both strategies based on
average pixel count and the current Observation's pixel count and employing different
strategies for observations that had very low black pixels VS observations that have
a substantial amount of black pixels.

2. Ended up calculating and using the error/noise level based on the naive bayes
prediction, but this does not seem to have much of an effect, in fact going by a
constant Probability of .99 and .01 seems to perform well in general as well.

3. Emission probabilities are normalized on the pixel counts, for HMM-VE every column
is scaled to prevent underflow and for Viterbi, log scaling is used to prevent underflow

Interesting Observations based on test images provided:

HMM-VE and HMM-MAP outperform Simple in almost every single test image provided.
When looking at HMM-VE vs HMM-MAP, the output is almost the same in terms of the prediction.
In fact HMM-VE has a better prediction of the last character, which I believe is due to the
Forward-Backward algorithm taking into consideration the end probabilities, which HMM-MAP
does not consider.

'''

from PIL import Image, ImageDraw, ImageFont
from naivebayes import NaiveBayes
from collections import defaultdict
from itertools import chain
from math import log
import sys

ch_width=14
ch_height=25

train_letters_ch = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
start_prob = defaultdict(int)
trans_prob = dict()
prior_prob = dict()
emm_prob = dict()
end_prob = defaultdict(int)


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


'''
Calls the NaiveBayes class 
provided in a separate file naivebayes.py
'''
def simplified_bayes(train_letters, test_letters, prior):
    nb = NaiveBayes(train_letters, prior)
    return ''.join([nb.predict(letter) for letter in test_letters])


'''
VE implementation using forward backward algorithm.
'''
def hmm_ve(test_letters):
    return forward_backward(test_letters, train_letters_ch)


'''
The forward backward algorithm to implement Variable Elimination,
the outer stub is taken from wikipedia - 
https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
and modified to fit the current problem.
'''
def forward_backward(observations, states):
    # Calculate the forward probabilities of being in a state Qn given all the
    # observations till On.
    fwd = []
    f_prev = {}
    for i, observation_i in enumerate(observations):
        f_curr = {}
        for st in states:
            if i == 0:
                prev_f_sum = start_prob[st]
            else:
                prev_f_sum = sum(f_prev[k] * trans_prob[k][st] for k in states)
            f_curr[st] = emm_prob[st][i] * prev_f_sum

        # Scale the column to sum to 1
        column_sum = sum(f_curr[st] for st in states)
        for st in states:
            f_curr[st] = f_curr[st]/column_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k] * end_prob[k] for k in states)

    # Calculate the backward probabilities of the observations On-1...Ok, given the state Qn
    bkw = []
    b_prev = {}
    
    for i in reversed(list(range(len(observations)))[1:] + [0]):
        b_curr = {}
        for st in states:
            if i == 0:
                b_curr[st] = end_prob[st]
            else:
                b_curr[st] = sum(trans_prob[st][l] * emm_prob[l][i] * b_prev[l] for l in states)

        # Scale the column to sum to 1
        column_sum = sum(b_curr[st] for st in states)
        for st in states:
            b_curr[st] = b_curr[st]/column_sum

        bkw.insert(0, b_curr)
        b_prev = b_curr

    posterior = []
    for i in range(len(observations)):
        # Prevent underflow, This should not happen, as we have scaled the values, just in case.
        if p_fwd == 0:
            return fwd, bkw, fwd
        else:
            posterior.append({st: fwd[i][st] * bkw[i][st] / p_fwd for st in states})

    return fwd, bkw, posterior


'''
HMM-MAP using the Viterbi implementation
'''
def hmm_map(test_letters):
    return viterbi(train_letters_ch, len(test_letters))


'''
The Viterbi implementation, 
builds a dp table with states as rows and observations as columns.
'''
def viterbi(states, num_obs):
    dp = {st: {obs: {} for obs in range(num_obs)} for st in states}
    start_prob_log = {st: log(start_prob[st]) if start_prob[st] > 0 else -100000 for st in states}
    trans_prob_log = {st: {q: log(trans_prob[st][q]) if trans_prob[st][q] > 0 else -100000 for q in states} for st in states}
    emm_prob_log = {st: {obs: log(emm_prob[st][obs]) if emm_prob[st][obs] > 0 else -100000 for obs in range(num_obs)} for st in states}

    for st in states:
        dp[st][0]['value'] = start_prob_log[st] + emm_prob_log[st][0]
        dp[st][0]['prev'] = None

    for obs in range(1, num_obs):
        for st in states:
            max_prev = max(dp[prev][obs-1]['value'] + trans_prob_log[prev][st] for prev in states)
            for prev in states:
                if dp[prev][obs-1]['value'] + trans_prob_log[prev][st] == max_prev:
                    dp[st][obs]['value'] = max_prev + emm_prob_log[st][obs]
                    dp[st][obs]['prev'] = prev
                    break

    last_st = None
    maxv = - float('inf')
    for st in states:
        if dp[st][num_obs-1]['value'] > maxv:
            maxv = dp[st][num_obs-1]['value']
            last_st = st
    res = [last_st]

    prev = last_st
    for obs in range(num_obs-1, 0, -1):
        st = dp[prev][obs]['prev']
        res = [st] + res
        prev = st
    return res


'''
Calculates the starting probabilities, 
end probabilities, transition probabilities
'''
def calculate_probabilities(fname):
    train_set = set(train_letters_ch)
    for ch1 in train_letters_ch:
        start_prob[ch1] = 1.0
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
                        start_prob[ch] += 1
                    elif j == len(line) - 1:
                        if i != len(lines)-1:
                            trans_prob[ch]['.'] += 1
                            end_prob['.'] += 1
                        end_prob[ch] += 1
                    elif line[j-1] in train_set:
                        trans_prob[line[j - 1]][ch] += 1

    initial_total = sum(start_prob.values())
    prior_total = sum(prior_prob.values())
    end_total = sum(end_prob.values())
    for ch1 in train_letters_ch:
        start_prob[ch1] = start_prob[ch1] / initial_total
        prior_prob[ch1] = prior_prob[ch1] / prior_total
        end_prob[ch1] = end_prob[ch1] / end_total
        trans_total = sum(trans_prob[ch1].values())
        for ch2 in train_letters_ch:
            trans_prob[ch1][ch2] = trans_prob[ch1][ch2] / trans_total
    end_prob['.'] = 0.1


'''
Calculates an approximate error/noise level, 
based on the prediction from Naive Bayes.
'''
def calculate_error(train_letters, test_letters, naive_prediction):
    total_error = 1
    total_valid = 1
    for i, ch in enumerate(naive_prediction):
        for j, pixel in enumerate(train_letters[ch]):
            if test_letters[i][j] == '*':
                total_error += (1 if pixel != test_letters[i][j] else 0)
                total_valid += (1 if pixel == test_letters[i][j] else 0)

    error_weight = 0.2 # Otherwise the Observation can get completely ignored, if naive bayes prediction is bad
    error_prob = error_weight * total_error / (total_error + total_valid)
    return error_prob


'''
Calculates the emission probabilities 
based on pixels and normalize them.
'''
def calculate_emission_prob(train_letters, test_letters, error_prob):
    pixel_count = {}
    for ch in train_letters_ch:
        total_on = 0
        for pix in train_letters[ch]:
            total_on += 1 if pix == '*' else 0
        pixel_count[ch] = total_on
    avg_pixel_count = sum(pixel_count.values())/len(train_letters_ch)
    for ch in train_letters_ch:
        emm_prob[ch] = [1.0] * len(test_letters)
        for i in range(len(test_letters)):
            total_on = 0
            for pix in test_letters[i]:
                total_on += 1 if pix == '*' else 0
            for j, pix in enumerate(test_letters[i]):
                if total_on > avg_pixel_count/5:
                    if pix == '*':
                        emm_prob[ch][i] *= (1 - error_prob) if pix == train_letters[ch][j] else error_prob
                else:
                    if pix == ' ':
                        emm_prob[ch][i] *= (1 - error_prob) if pix == train_letters[ch][j] else error_prob

    for i in range(len(test_letters)):
        total = 0
        for ch in train_letters_ch:
            total += emm_prob[ch][i]
        for ch in train_letters_ch:
            emm_prob[ch][i] = emm_prob[ch][i]/total


def main():
    #train_img_fname = 'courier-train.png'
    #train_txt_fname = 'DemocracyAndEducation.txt'

    if len(sys.argv) < 4:
        print('Usage: ')
        print('./ocr.py train-image-file.png train-text.txt test-image-file.png')
        sys.exit()

    (train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:4]

    train_letters = load_training_letters(train_img_fname)
    for ch in train_letters_ch:
        train_letters[ch] = list(chain.from_iterable(train_letters[ch]))
    calculate_probabilities(train_txt_fname)

    for i in range(1):
        #test_img_fname = 'test-{}-0.png'.format(i)
        test_letters = load_letters(test_img_fname)
        for i, l in enumerate(test_letters):
            test_letters[i] = list(chain.from_iterable(test_letters[i]))

        # Simplified
        simplified_res = simplified_bayes(train_letters, test_letters, prior_prob)
        print(' Simple: {}'.format(simplified_res))

        calculate_emission_prob(train_letters, test_letters, calculate_error(train_letters, test_letters, simplified_res))

        # HMM VE
        fwd, bkw, posterior = hmm_ve(test_letters)
        print(' HMM VE: {}'.format(''.join([max(test_prob, key=test_prob.get) for test_prob in posterior])))

        # HMM MAP
        print('HMM MAP: {}'.format(''.join(hmm_map(test_letters))))


if __name__ == '__main__':
    main()
