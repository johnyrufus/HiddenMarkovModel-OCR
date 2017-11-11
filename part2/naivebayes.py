#!/usr/bin/env python3

import copy
import math

'''
A simple Naive Bayes model to predict OCR characters
'''
class NaiveBayes:
    def __init__(self, train_data, prior):
        self.train_data = train_data
        # Assume uniform priors for now, can be changed to reflect actual priors based on some general distribution
        if prior is None:
            prior = {'A':1, 'B':1, 'C':1, 'D':1, 'E':1, 'F':1, 'G':1, 'H':1, 'I':1, 'J':1, 'K':1, 'L':1, 'M':1, 'N':1, 'O':1, 'P':1, 'Q':1, 'R':1, 'S':1, 'T':1, 'U':1, 'V':1, 'W':1, 'X':1, 'Y':1, 'Z':1,
                      'a':1, 'b':1, 'c':1, 'd':1, 'e':1, 'f':1, 'g':1, 'h':1, 'i':1, 'j':1, 'k':1, 'l':1, 'm':1, 'n':1, 'o':1, 'p':1, 'q':1, 'r':1, 's':1, 't':1, 'u':1, 'v':1, 'w':1, 'x':1, 'y':1, 'z':1,
                      '0':1, '1':1, '2':1, '3':1, '4':1, '5':1, '6':1, '7':1, '8':1, '9':1,
                      '(':1, ')':1, ',':1, '.':1, '-':1, '!':1, '?':1, '"':1, '\'':1, ' ':1}
        self.prior = prior

    '''
    Predicts the values based on training data, and binary features based on pixels, 
    0 and 1 values are laplace smoothed and transformed to 1/3 and 2/3 respectively.
    '''
    def predict(self, test_data):
        posterior = copy.deepcopy(self.prior)
        for label in posterior.keys():
            res = math.log(self.prior[label])
            for i, val in enumerate(test_data):
                if test_data[i] == '*':
                    res += math.log(2 / 3) if self.train_data[label][i] == test_data[i] else math.log(1 / 3)
            posterior[label] = res
        return max(posterior, key=posterior.get)


