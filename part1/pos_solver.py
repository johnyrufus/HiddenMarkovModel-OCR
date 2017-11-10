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
from enum import Enum
from math import log

class Algo(Enum):
    Simplified = "Simplified"
    HMM_VE = "HMM VE"
    HMM_MAP = "HMM MAP"

class Cell():
    '''
    Represents a cell in a Viterbi matrix. Has two members:
        * prevPos - the cell in the previous state that is the most likely predecessor of this cell
        * prob - the joint distribution probability of this cell
    
    Cells in the initial state should have a prevPos of None.
    
    When the last state has in the Viterbi matrix has been evaluated, the maximum
    probability POS for that state can be determined. Then you only need to follow
    the prevPos links until a cell with prevPos == None is reached, and you will
    have your MAP path.
    '''
    
    def __init__(self, prevPos, prob):
        self.prevPos = prevPos
        self.prob = prob


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    order = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", \
             "verb", "x", "."]

    def __init__(self):
        #Class variables for the simplified POS model.
        self.simple_p_of_pos_given_word = {} # key = word, value = list of 12 POS probabilities
        self.simple_p_of_pos = [0.00] * 12 # list of 12 POS probabilities

        #Class variables for the HMM-VE model.
        self.initial_state_probs = [0.00] * 12
        self.end_state_probs = [0.00] * 12
        self.emission_probs = [Counter() for _ in range(12)]
        self.transition_probs = [[0.00] * 12 for _ in range(12)]
        self.lexicon = Counter()
        
        self.posteriors = {Algo.Simplified: 0.0, Algo.HMM_VE: 0.0, Algo.HMM_MAP: 0.0}

        #Class variables for the HMM-Viterbi model.
        self.initial_state_probs_log = [0.00] * 12
        self.end_state_probs_log = [0.00] * 12
        self.emission_probs_log = [{} for _ in range(12)]
        self.transition_probs_log = [[0.00] * 12 for _ in range(12)]
            
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, algo):
        return self.posteriors[algo]

    # Do the training!
    #
    def train(self, data):
        for example in data:
            sentence = example[0]
            tags = example[1]

            #Variables for VE algorithm.
            #Count the times a POS starts a sentence.
            self.initial_state_probs[self.order.index(tags[0])] += 1
            self.end_state_probs[self.order.index(tags[-1])] += 1
            
            self.initial_state_probs_log[self.order.index(tags[0])] += 1
            self.end_state_probs_log[self.order.index(tags[-1])] += 1            
            
            last_tag = None
            for word, tag in zip(sentence, tags):
                index_of_tag = self.order.index(tag)
                
                #Variables for simplified algorithm.
                self.simple_p_of_pos[index_of_tag] += 1
                if word not in self.simple_p_of_pos_given_word:
                    self.simple_p_of_pos_given_word[word] = [0.00] * 12
                self.simple_p_of_pos_given_word[word][index_of_tag] += 1
        
                # Transition Probabilities
                if last_tag != None:
                    last_tag_index = self.order.index(last_tag)
                    current_tag_index = self.order.index(tag)
                    
                    self.transition_probs[last_tag_index][current_tag_index] += 1

                # Emission Probabilities
                self.emission_probs[index_of_tag][word] += 1
                
                if word not in self.emission_probs_log[index_of_tag]:
                    self.emission_probs_log[index_of_tag][word] = 0
                self.emission_probs_log[index_of_tag][word] += 1

                # Word probabilities                
                self.lexicon[word] += 1
                
                last_tag = tag
                
        #Converting P(pos) to percentages
        total = sum(self.simple_p_of_pos)
        for p, pos in enumerate(self.simple_p_of_pos):
            self.simple_p_of_pos[p] = pos / total
        
        #Converting P(pos|word) to percentages
        for word in self.simple_p_of_pos_given_word:
            probs = self.simple_p_of_pos_given_word[word]
            
            total = sum(probs)
            probs = [x / total for x in probs]
            
            self.simple_p_of_pos_given_word[word] = probs

        #Converting initial states to percentages
        total = sum(self.initial_state_probs)
        self.initial_state_probs = [x / total for x in self.initial_state_probs]
        
        total = sum(self.initial_state_probs_log)
        self.initial_state_probs_log = [log(x / total, 2) for x in self.initial_state_probs_log]
        
        total = sum(self.end_state_probs)
        self.end_state_probs = [x / total for x in self.end_state_probs]

        #Converting transitions over to percentages:
        for p, pos in enumerate(self.transition_probs):
            total = sum(pos)
            probs = [(x+1.00) / (total+12.00) for x in pos]
            self.transition_probs[p] = probs
            
        for p, pos in enumerate(self.transition_probs_log):
            total = sum(pos)
            probs = [log((x + 1.00) / (total + 12.00), 2) for x in pos]

        #Converting emission probabilities over to percentages:
        for p, pos in enumerate(self.emission_probs):
            total = sum(pos.values())
            
            for word in pos:
                pos[word] = 1.00 * pos[word] / total
        
        for p, pos in enumerate(self.emission_probs_log):
            total = sum(pos.values())
            
            for word in pos:
                pos[word] = log(1.00 * pos[word] / total, 2)
    
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        guess = []
        s_posterior = 1.0        
        for word in sentence:
            if word in self.simple_p_of_pos_given_word:
                probs = self.simple_p_of_pos_given_word[word]
                maxProb = max(probs)
                part = probs.index(maxProb)                
            else:
                #If we've never encountered the word before, pick the most
                #common part-of-specch.
                maxProb = max(self.simple_p_of_pos)
                part = self.simple_p_of_pos.index(maxProb)
            guess += [self.order[part]]
            s_posterior *= maxProb
        self.posteriors[Algo.Simplified] = log(s_posterior)
        return guess
    
    #Shell of the code: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
    #Adjusted to suit our problem/formulation.
    def hmm_ve(self, sentence):
        guess = []
        #Calculate foward pass...
        forward = []
        ve_posterior = 1.0
        prev_state = [0.00] * 12
        for w, word in enumerate(sentence):
            current_state = [0.00] * 12
            
            for c, c_pos in enumerate(current_state):
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
            maxProb = max(probs)
            ve_posterior *= maxProb
            guess += [self.order[probs.index(maxProb)]]

        self.posteriors[Algo.HMM_VE] = log(ve_posterior)
        return guess

    def hmm_viterbi(self, sentence):
        guess = []
        
        state = [[0.00] * 12 for x in range(len(sentence))]
        backtrack = []

        vit_posterior = 1.0
        
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

        vit_posterior = max(last)

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

        self.posteriors[Algo.HMM_MAP] = vit_posterior
        return guess

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == Algo.Simplified:
            return self.simplified(sentence)
        elif algo == Algo.HMM_VE:
            return self.hmm_ve(sentence)
        elif algo == Algo.HMM_MAP:
            return self.hmm_viterbi(sentence)
        else:
            raise ValueError("Unknown algo")
