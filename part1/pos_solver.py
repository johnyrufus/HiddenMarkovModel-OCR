###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids: Johny Rufus (johnjoh), Chris Falter (cfalter), Anthony Duer (anduer)
#
# (Based on skeleton code by D. Crandall)
#
#
####
# The program below implements three versions of the Hidden Markov Model 
# given to us a part of Assignment 3, Part 1. In total, we implmeneted four
# features that fully develop the model and then implement inference on it.
# 
# The first section we tackled was to implemnet the training method. It is
# responsible for calculating the various pieces of the puzzle that we need
# in order to implement all three algorithms:
# 
# Simplified - requires P(Part-of-Speech | Word) and is implemented as d
# dictionary of words with each word pointing to a list with 12 components. 
# Those components then contain the probability of a part-of-speech based off
# of the training data. We also implemented a raw P(Part-of-Speech) for words
# we have never seen before. This simply returns the most common part-of-specch
# (noun) that we did not see while training.
# 
# Hidden Markov Model - Variable Elimination (VE): this model requires initial
# state probabilities, end state probabilities, emission probabilities, and 
# transition probabilities. The initial and end state probabilities are both
# calculated the same - any time a part-of-speech is at the front or end of a
# sentence, then the count for that part-of-speech is incremented by 1. 
# Likewise, for each word encountered, we add it to a dictionary of words for
# each part-of-speech and to our lexicon (if it doesn't already exist). We then
# increment the count by one. Finally, for all words beyond the first, we 
# record the last and current part-of-speech tag and then increment by one. 
# Once training is completed, each variable is normalized to 1.
# 
# Viterbi Decoding: This algorithm requires the same as HMM-VE but, instead of
# decimal probabilities, each value is expressed in terms of log base 2 to
# avoid underflow issues.
# 
# During the development of this section, we did not encounter any significant
# difficulties other than simple programming errors (forgetting to normalize
# or convert to log base 2). There were some changes made to what structures
# were used to store the data but those were quickly settled upon after a few
# trials.
# 
# The second function we implemented was the Simplified method. This algorithm
# was fairly easy to implement - for every word, we simply picked the most 
# common part-of-speech. If we had never encountered the word before, we 
# guess by selecting the most common part-of-speech from our training dataset,
# which was noun. We found the power of even this rudimentary system to be 
# pretty good - around 94% on words and 47% on sentences. We did not have to
# do much tweaking to the algorithm once it was properly implemented.
#
# The third function implemented was the Variable Elimination Hidden Markov
# Model. This one took some amount of time to get right as we initially 
# implemented only a forward pass on the data which did not perform as well as 
# the simple method. However, postings on the Piazza board with Professor 
# Crandall pointed us in the right direction: to use the forward/backward 
# algorithm. Once implemented, we saw a significant boost to performance: 95%
# on words and 54.3% on sentences. We opted to not implement Laplace smoothing
# at this point as initial tests showed it to cause a slgiht regression in 
# performance.
# 
# The four fucntion implemented was the Viterbi algorithm. This one, in its
# base form, came together fairly quickly as we had a very simple model working
# in Excel that was developed during the Viterbi module. We were able to use 
# that as a reference point to quickly implement and tune our part-of-speech
# tagging version. We were required to implement Laplace smoothing as log(0, 2)
# is undefined. After a bit of bug hunting and refinements, we were happy with
# our implemtntation which showed a marginal regression in word perfromance
# when compared to HMM-VE but that was more than offset by the improvement on
# overall sentences (0.03% drop on words, 0.15% improvement on sentences). This
# leads to speculation that when it gets things wrong, it tends to produce 
# incorrect sequences (multiple words wrong per sentence rather than just one 
# word) but we did not investigate the issue further.
#
# The final function implemented was the posterior distribution. We did not 
# have a good intuition about it to begin with as it wasn't heavily covered
# in the lectures. Our initial guess was to implement it differently for each
# algorithm based off of how they model the world. For example, we assumed that
# the simplified method's posterior was simply the probability of generating
# that sequence at random (randomly pick one of 12 part-of-speech tags N times
# over where N is the length of the sentence) which became (1/12)^N. Likewise,
# the HMM-VE version used P(S)P(S2|S)P(S3|S2)...P(Sn|Sn-1). This was close to
# the correct method, which we have implemented below, that was discussed on
# Piazza by Professor Crandall after we asked a question about it.
#
#References:
#HMM-VE shell code was adapted from the Wikipedia article on the algorithm:
#https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

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
    #This is the sequence that we stored all tags in for lists.
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
        #The lexicon is the number of unique words encountered. We used this 
        #for Laplace smoothing as the denominator.
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
        #This is where we calculate the probability of the tag sequence.
        #P(S)P(S2 | S)P(S3 | S2)...
        for t, tag in enumerate(tags):
            tag_index = self.order.index(tag)
            if t == 0: posterior += self.initial_state_probs_log[tag_index]
            else: posterior += self.transition_probs_log[last_tag_index][tag_index]            
            last_tag_index = tag_index
            
        #This is where we calculate the probability of the emissions given the 
        #tags.
        #P(W1|S1)P(W2|S2)P(W3|S3)...
        for word, tag in zip(sentence, tags):
            tag_index = self.order.index(tag)
            if word in self.emission_probs_log[tag_index]:
                posterior += self.emission_probs_log[tag_index][word]
            else:
                posterior += log(1.00 / (len(self.lexicon)*10000), 2)
        
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
