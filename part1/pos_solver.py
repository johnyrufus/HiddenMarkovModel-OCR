###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math

order = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", \
         "verb", "x", "."]


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        self.simp_dict = {}
        self.locations = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        #Simplified Dictionary Creation
        for example in data:           
            words = example[0]
            tags = example[1]
        
            for word, tag in zip(words, tags):
                if word not in self.simp_dict:
                    self.simp_dict[word] = [0] * 12
            
                self.simp_dict[word][order.index(tag)] += 1
        pass

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        guess = []
        
        for word in sentence:
            if word in self.simp_dict:
                values = self.simp_dict[word]
                part = values.index(max(values))
                
                guess += [order[part]]
            else:
                guess += ["noun"]
        return guess

    def hmm_ve(self, sentence):
        return [ "noun" ] * len(sentence)

    def hmm_viterbi(self, sentence):
        return [ "noun" ] * len(sentence)


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"

