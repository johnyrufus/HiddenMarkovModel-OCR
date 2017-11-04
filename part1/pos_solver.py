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
from copy import deepcopy

order = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", \
         "verb", "x", "."]

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        #Class variables for the simplified POS model.
        self.emissions = {}
        self.totals = [0.00] * 12
        
        #Class variables for the HMM-VE model.
        self.initial_state = [0.00] * 12
        self.transitions = [[0.00] * 12 for x in range(12)]
        
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        #Simplified Dictionary Creation
        for example in data:           
            sentence = example[0]
            tags = example[1]
            
            index_of_tag = order.index(tags[0])
            self.initial_state[index_of_tag] += 1
        
            last_tag = None
            for word, tag in zip(sentence, tags):
                if word not in self.emissions:
                    self.emissions[word] = [0.00] * 12
            
                index_of_tag = order.index(tag)
                self.emissions[word][index_of_tag] += 1
                self.totals[index_of_tag] += 1
                
                if last_tag != None:
                    index_of_last = order.index(last_tag)
                    self.transitions[index_of_last][index_of_tag] += 1
                last_tag = tag
        
        #Convert initial states to percentages
        total = sum(self.initial_state)
        self.initial_state = [1.00 * x / total for x in self.initial_state]
 
        #Convert emission probabilities over to percentages.
        for key in self.emissions:
            values = self.emissions[key]
            total = sum(values)
            
            values = [1.00 * x / total for x in values]
            self.emissions[key] = values     
        
        #Convert transitions over to percentages...
        for i, row in enumerate(self.transitions):
            total = sum(row)
            row = [1.00 * x / total for x in row]
            self.transitions[i] = row
        pass
    
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        guess = []
        
        for word in sentence:
            if word in self.emissions:
                values = self.emissions[word]
                part = values.index(max(values))
                
                guess += [order[part]]
            else:
                #If we've never encountered the word before, pick the most
                #common part-of-specch.
                part = self.totals.index(max(self.totals))
                guess += [order[part]]
        return guess

    def hmm_ve(self, sentence):
        guess = []
        
        state = [0.00] * 12
        for i, word in enumerate(sentence):            
            if i == 0:
                for y, pos in enumerate(state):
                    pos = self.initial_state[y]

                    #If we've never encountered this word before, then
                    #let everything have equal probability of emission.
                    if word in self.emissions:
                        pos *= self.emissions[word][y]
                    state[y] = pos
            else:
                new_state = [0.00] * 12
                for n, new in enumerate(new_state):
                    for p, pos in enumerate(state):
                        trans_prob = self.transitions[p][n]
                        prev_prob = state[p]
                        
                        new_state[n] += trans_prob * prev_prob
                    
                    #If we've never encountered this word before, then
                    #let everything have equal probability of emission.
                    if word in self.emissions:
                        new_state[n] *= self.emissions[word][n]
                state = new_state
            
            total = sum(state)
            state = [1.00 * x / total for x in state]
            
            part = state.index(max(state))
            guess += [order[part]]
        
        return guess

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

