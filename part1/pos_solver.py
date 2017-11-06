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
import math

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

order = ["adj", "adv", "adp", "conj", "det", "noun", "num", "pron", "prt", \
         "verb", "x", "."]

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        #Class variables for the simplified POS model.
        self.simple_p_of_pos_given_word = {}
        self.simple_p_of_pos = [0.00] * 12  

        #Class variables for the HMM-VE model.
        self.hmm_emissions = [Counter() for _ in range(12)]
        self.hmm_initial_state = [0.00] * 12
        self.hmm_transitions = [[0.00] * 12 for _ in range(12)]
        ## Clean-up #self.hmm_default = [0.00] * 12
        self.hmm_lexicon = Counter()

#        #Class variables for the HMM-Viterbi model.
#        self.emissions_log = {}
#        self.initial_state_log = [0.00] * 12
#        self.transitions_log = [[0.00] * 12 for x in range(12)]
#        self.default_log = [0.00] * 12
        
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        #Count how many times a tag starts a sentence
        
        for example in data:
            sentence = example[0]
            tags = example[1]

            #Variables for VE algorithm.
            #Count the times a POS starts a sentence.
            self.hmm_initial_state[order.index(tags[0])] += 1
            
            last_tag = None
            for word, tag in zip(sentence, tags):
                index_of_tag = order.index(tag)
                
                #Variables for simplified algorithm.
                self.simple_p_of_pos[index_of_tag] += 1
                if word not in self.simple_p_of_pos_given_word:
                    self.simple_p_of_pos_given_word[word] = [0.00] * 12
                self.simple_p_of_pos_given_word[word][index_of_tag] += 1
        
                # Transition Probabilities
                if last_tag != None:
                    last_tag_index = order.index(last_tag)
                    current_tag_index = order.index(tag)
                    
                    self.hmm_transitions[last_tag_index][current_tag_index] += 1

                # Emission Probabilities
                self.hmm_emissions[index_of_tag][word] += 1

                # Word probabilities                
                self.hmm_lexicon[word] += 1
                
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
        total = sum(self.hmm_initial_state)
        self.hmm_initial_state = [x / total for x in self.hmm_initial_state]

        #Converting transitions over to percentages:
        for p, pos in enumerate(self.hmm_transitions):
            total = sum(pos)
            probs = [x / total for x in pos]
            self.hmm_transitions[p] = probs

        #Converting emission probabilities over to percentages:
        for p, pos in enumerate(self.hmm_emissions):
            total = sum(pos.values())
            
            for word in pos:
                pos[word] = 1.00 * pos[word] / total
            
            #Record our probability of never having encountered a word.
            ## Clean-up #self.hmm_default[p] = 1.00 / total
    
    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        guess = []        
        for word in sentence:
            if word in self.simple_p_of_pos_given_word:
                probs = self.simple_p_of_pos_given_word[word]
                part = probs.index(max(probs))
                
                guess += [order[part]]
            else:
                #If we've never encountered the word before, pick the most
                #common part-of-specch.
                part = self.simple_p_of_pos.index(max(self.simple_p_of_pos))
                guess += [order[part]]
        return guess

    def hmm_ve(self, sentence):
        guess = []
        
        state = [0.00] * 12
        for w, word in enumerate(sentence):
            if w == 0:
                for p, pos in enumerate(state):
                    prob = self.hmm_initial_state[p]
                    
                    if word in self.hmm_lexicon:
                        # word is in lexicon, so it must be in emissions
                        prob *= self.hmm_emissions[p][word] 
                    state[p] = prob
            else:
                new_state = [0.00] * 12
                for new_p, new_pos in enumerate(new_state):
                    for old_p, old_pos in enumerate(state):
                        trans_prob = self.hmm_transitions[old_p][new_p]                        
                        new_state[new_p] += trans_prob * old_pos
                    #If we've never encountered this word before, then
                    #let everything have equal probability of emission.
                    if word in self.hmm_lexicon:
                        # word is in lexicon, so it must be in emissions
                        new_state[new_p] *= self.hmm_emissions[new_p][word]
                state = new_state
            
            total = sum(state)
            state = [1.00 * x / total for x in state]

            part_index = state.index(max(state))
            guess += [order[part_index]]   
        return guess

    def hmm_viterbi(self, sentence):
        guess = ["noun"] * len(sentence)
#        
#        state = [[0.00] * 12 for x in range(len(sentence))]
#        backtrack = []
#        
#        for w, word in enumerate(sentence):
#            if word in self.emissions_log:
#                em = self.emissions_log[word]
#            else:
#                em = self.default_log
#            
#            if w == 0:
#                cur_state = state[w]
#                
#                for c, c_pos in enumerate(cur_state):
#                    value = self.initial_state_log[c] + em[c]
#                    cur_state[c] = value
#                state[w] = cur_state
#            else:
#                prev_state = state[w-1]
#                cur_state = state[w]
#                
#                for c, c_pos in enumerate(cur_state):
#                    prev_val = -float('inf')
#                    for p, p_pos in enumerate(prev_state):
#                        prev_val = max(prev_val, p_pos + self.transitions_log[p][c])
#                    
#                    value = em[c] + prev_val
#                    cur_state[c] = value
#                state[w] = cur_state
#
#                back = [[0.00] * 12 for x in range(12)]
#                for c, c_pos in enumerate(cur_state):
#                    for p, p_pos in enumerate(prev_state):
#                        back[p][c] = p_pos + self.transitions_log[p][c] + em[c]
#                backtrack.append(back)
#
#        last = state[len(sentence)-1]
#        last_index = last.index(max(last))
#        tagging = [last_index]
#
#        for trans in reversed(backtrack):
#            high_prob = -float('inf')
#            index = -1
#            
#            for p, pos in enumerate(trans):
#                if pos[last_index] > high_prob:
#                    high_prob = pos[last_index]
#                    index = p
#            
#            last_index = index
#            tagging = [last_index] + tagging
#
#        for tag in tagging:
#            guess += [order[tag]]

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
            print ("Unknown algo!")
