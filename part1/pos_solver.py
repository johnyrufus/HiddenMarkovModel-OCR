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

from collections import Counter
from math import log
import numpy as np

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

safeLog = np.vectorize(lambda p: log(p, 2) if p > 0 else -10000.0, otypes=[np.float])

class Solver:
    order = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", \
             "verb", "x", "."]

    def __init__(self):
        #Class variables for the simplified POS model.
        self.simple_p_of_pos_given_word = {} # key = word, value = list of 12 POS probabilities
        self.simple_p_of_pos = np.array([0.00] * 12) # list of 12 POS probabilities

        #Class variables for the HMM-VE model.
        self.initial_state_probs = np.array([0.00] * 12)
        self.end_state_probs = np.array([0.00] * 12)
        self.emission_probs = [Counter() for _ in range(12)]
        self.transition_probs = np.array([[0.00] * 12 for _ in range(12)])
        self.lexicon = Counter()
        
        #Class variables for the HMM-Viterbi model.
        self.initial_state_probs_log = np.array([0.00] * 12) # will be set after training
        self.end_state_probs_log = np.array([0.00] * 12) # will be set after training
        self.emission_probs_log = [{} for _ in range(12)]
        self.transition_probs_log = np.array([[0.00] * 12 for _ in range(12)]) # will be set after training
            
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, tags):
        posterior = 0
        
        last_tag_index = None
        for t, tag in enumerate(tags):
            tag_index = self.order.index(tag)
            if t == 0: posterior += self.initial_state_probs_log[tag_index]
            else: posterior += self.transition_probs_log[last_tag_index][tag_index]            
            last_tag_index = tag_index

        for word, tag in zip(sentence, tags):
            tag_index = self.order.index(tag)
            if word in self.emission_probs_log[tag_index]:
                posterior += self.emission_probs_log[tag_index][word]
            else:
                posterior += log(1.00 / (len(self.lexicon)*10000), 2)
        
        #P(W1 | S1) ... P(Wn | Sn) P(S1) P(S2|S1) ... P(Sn | Sn-1)/ P(W1...Wn)
        return posterior

    # Do the training!
    #
    def train(self, data):
        for example in data:
            sentence = example[0]
            tags = example[1]

            #Variables for VE algorithm.
            #Count the times a POS starts a sentence.
            self.initial_state_probs[self.order.index(tags[0])] += 1.
            self.end_state_probs[self.order.index(tags[-1])] += 1.
            
            last_tag = None
            for word, tag in zip(sentence, tags):
                index_of_tag = self.order.index(tag)
                
                #Variables for simplified algorithm.
                self.simple_p_of_pos[index_of_tag] += 1.
                if word not in self.simple_p_of_pos_given_word:
                    self.simple_p_of_pos_given_word[word] = np.array([0.00] * 12)
                self.simple_p_of_pos_given_word[word][index_of_tag] += 1.
        
                # Transition Probabilities
                if last_tag != None:
                    last_tag_index = self.order.index(last_tag)
                    current_tag_index = self.order.index(tag)
                    
                    self.transition_probs[last_tag_index][current_tag_index] += 1.
                    self.transition_probs_log[last_tag_index][current_tag_index] += 1

                # Emission Probabilities
                self.emission_probs[index_of_tag][word] += 1.

                if word not in self.emission_probs_log[index_of_tag]:
                    self.emission_probs_log[index_of_tag][word] = 0
                self.emission_probs_log[index_of_tag][word] += 1
                
                # Word probabilities                
                self.lexicon[word] += 1.
                
                last_tag = tag
                
        #Converting P(pos) to percentages
        total = np.sum(self.simple_p_of_pos)
        self.simple_p_of_pos /= total
        
        #Converting P(pos|word) to percentages
        for word in self.simple_p_of_pos_given_word:
            probs = self.simple_p_of_pos_given_word[word]            
            total = sum(probs)
            probs /= total            

        #Converting initial states to percentages
        total = sum(self.initial_state_probs)
        self.initial_state_probs /= total
        self.initial_state_probs_log = safeLog(self.initial_state_probs)
        
        total = sum(self.end_state_probs)
        self.end_state_probs /= total
        self.end_state_probs_log = safeLog(self.end_state_probs)

        #Converting transitions over to percentages:
        # Note that Laplace smoothing has been removed
        self.transition_probs = np.apply_along_axis(
                lambda row: row/np.sum(row), 1, self.transition_probs)
        self.transition_probs_log = safeLog(self.transition_probs)
            
        #Converting emission probabilities over to percentages:
        for i, pos in enumerate(self.emission_probs):
            total = sum(pos.values())            
            for word in pos:
                pos[word] /= total
                self.emission_probs_log[i][word] = log(pos[word], 2)
            
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        guess = []
        for word in sentence:
            if word in self.simple_p_of_pos_given_word:
                probs = self.simple_p_of_pos_given_word[word]
            else:
                #If we've never encountered the word before, pick the most
                #common part-of-specch.
                probs = self.simple_p_of_pos
            part = np.argmax(probs)               
            guess += [self.order[part]]
        return guess
    
    #Shell of the code: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    #Adjusted to suit our problem/formulation.
    def hmm_ve(self, sentence):
        guess = []
        #Calculate foward pass...
        forward = []
        prev_state = [0.00] * 12
        for w, word in enumerate(sentence):
            current_state = [0.00] * 12
            
            for c in range(len(current_state)):
                #We're at the start of the sentence, so use initial_states...
                if w == 0:
                    prob = self.initial_state_probs[c]
                #Otherwise calculate the sum of our transitions...
                else:
                    prob = 0
                    for p, p_pos in enumerate(prev_state):
                        prob += prev_state[p] * self.transition_probs[p][c]
                
                current_state[c] = prob
                #If our word is in our lexicon...
                if word in self.lexicon:
                    current_state[c] *= self.emission_probs[c][word]
            forward.append(current_state)
            prev_state = current_state
        forward_prob = sum(current_state[p] * self.end_state_probs[p] for p in range(12))
        
        backward = []
        prev_state = [0.00] * 12

        for w, word in enumerate(reversed(sentence[1:] + ("",))):
            current_state = [0.00] * 12
            
            for c, c_pos in enumerate(current_state):
                if w == 0:
                    current_state[c] = self.end_state_probs[c]
                else:
                    prob = 0.00
                    for p, p_pos in enumerate(prev_state):     
                        temp = self.transition_probs[c][p] * prev_state[p]
                        if word in self.lexicon:
                            temp *= self.emission_probs[p][word]
                        prob += temp
                    current_state[c] = prob
            backward.insert(0, current_state)
            prev_state = current_state
        
        for w in range(len(sentence)):
            probs = [forward[w][pos] * backward[w][pos] / forward_prob for pos in range(12)]
            pos_index = probs.index(max(probs))    
            guess += [self.order[pos_index]]

        return guess

    def hmm_viterbi(self, sentence):
        guess = []
        
        guess = []
        
        state = [[0.00] * 12 for x in range(len(sentence))]
        backtrack = []
        
        for w, word in enumerate(sentence):
            if w == 0:
                cur_state = state[w]
                
                for c, c_pos in enumerate(cur_state):
                    value = self.initial_state_probs_log[c]
                    
                    if word in self.emission_probs_log[c]:
                        value += self.emission_probs_log[c][word]
                    else:
                        value += log(1.00 / (len(self.lexicon)*10000), 2)
                    cur_state[c] = value
                state[w] = cur_state
            else:
                prev_state = state[w-1]
                cur_state = state[w]
                
                for c, c_pos in enumerate(cur_state):
                    value = -float('inf')
                    for p, p_pos in enumerate(prev_state):
                        value = max(value, p_pos + self.transition_probs_log[p][c])

                    if word in self.emission_probs_log[c]:
                        value += self.emission_probs_log[c][word]
                    else:
                        value += log(1.00 / (len(self.lexicon)*100), 2)
                    cur_state[c] = value
                state[w] = cur_state

                back = [[0.00] * 12 for x in range(12)]
                for c, c_pos in enumerate(cur_state):
                    for p, p_pos in enumerate(prev_state):
                        back[p][c] = p_pos + self.transition_probs_log[p][c]
                        
                        if word in self.emission_probs_log[c]:
                            back[p][c] += self.emission_probs_log[c][word]
                        else:
                            back[p][c] += log(1.00 / (len(self.lexicon)*10000), 2)
                            
                backtrack.append(back)
                
        last = state[len(sentence)-1]
        last_index = last.index(max(last))
        tagging = [last_index]

        for trans in reversed(backtrack):
            high_prob = -float('inf')
            index = -1
            
            for p, pos in enumerate(trans):
                if pos[last_index] > high_prob:
                    high_prob = pos[last_index]
                    index = p
            last_index = index
            tagging = [last_index] + tagging

        for tag in tagging:
            guess += [self.order[tag]]

        return guess

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
            raise ValueError("Unknown algo")
