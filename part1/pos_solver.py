#!/usr/bin/python2
###################################
# CS B551 Fall 2017, Assignment #3
# D. Crandall
#
# There should be no need to modify this file, although you 
# can if you really want. Edit pos_solver.py instead!
#
# To get started, try running: 
#
#   python ./label.py bc.train bc.test.tiny
#

from pos_scorer import Score
from pos_solver import Algo, Solver
import sys

# Read in training or test data file
#
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]

    return exemplars


####################
# Main program
#

#if len(sys.argv) < 3:
#    print ("Usage: ")
#    print ("    ./label.py training_file test_file")
#    sys.exit()

#(train_file, test_file) = sys.argv[1:3]

train_file = "C:\\users\\aduer\\desktop\\part1\\bc.train.txt"
test_file = "C:\\users\\aduer\\desktop\\part1\\bc.test.txt"

print ("Learning model...")
solver = Solver()
train_data = read_data(train_file)
solver.train(train_data)

print ("Loading test data...")
test_data = read_data(test_file)

print ("Testing classifiers...")
scorer = Score()
Algorithm_labels = [str(i+1) + ". " + key for i, key in enumerate(Algo.__members__)]
gtLabel = "0. Ground truth"
for (s, gt) in test_data:
    outputs = {gtLabel : gt}
    posteriors = {gtLabel: 0.0}
        
    # run all algorithms on the sentence
    for (algoName, label) in zip(Algo.__members__, Algorithm_labels):
        outputs[label] = solver.solve(Algo[algoName], s) 
        posteriors[label] = solver.posterior(s, Algo[algoName])

    Score.print_results(s, outputs, posteriors)
        
    scorer.score(outputs)
    scorer.print_scores()
    
    print ("----")
