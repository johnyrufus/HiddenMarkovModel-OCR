# B551 Assignment 3: Probability
Anthony Duer, Johny Rufus, Chris Falter, Fall 2017

In the following discussion, we use these symbols:
+ P - probability
+ S - part of speech (adjective, adverb, adposition, conjunction, determiner, noun, number, pronoun, particle, verb, foreign word, or punctuation mark)
+ W - token (a word, number, punctuation mark, etc.)

## Part 1: Part-of-Speech Tagging
In part 1, we used a Hidden Markov Model (HMM) to infer the parts of speech for each word in a sentence. We compared three different HMM algorithms in their ability to predict both the POS of each individual word and the entire POS sequence for whole sentences. 

The three algorithms were:
1. **Simplified**: This algorithm predicts parts of speech based solely on the individual words in a sentence. It does not incorporate initial state probabilities for the first POS, end state probabilities for the last POS, or transition probabilities from one POS to another. The probability that a token W<sub>1</sub> represents a part of speech S<sub>1</sub> is simply P(S<sub>1</sub> | W<sub>1</sub>) as calculated from training data. 

    The algorithm selects the S with the maximum likelihood for any given W. If the W did not appear in the training data, the algorithm selects the most common S in the training data. In the training data provided in the assignment, the most common part of speech is a noun.

2. **HMM Variable Elimination (VE)**: This algorithm uses initial state probabilities, transition probabilities, and end state probabilities for parts of speech, in combination with emission probabilities of words given the part of speech. It estimates the probability of the part of speech that corresponds to a word by multiplying emission probabilities by the probabilities of the 12 parts of speech, which in turn are estimated based on the position of the word in the sentence (start or end) and, if applicable, the transition probabilities from the previous part of speech. We implement dynamic programming for a sentence of length *n* by storing the a list of 12 POS probabilities in a list of size *n*. 

    We initially implemented only a forward pass over the tokens in a sentence, but this yielded predictions that were less accurate than the Simplified algorithm's. Piazza discussions with Professor Crandall pointed us toward using the forward/backward algorithm. [Following the explanation on Wikipedia](https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm), two dynamic programming matrices (consisting of a list of lists) are created: forward (from beginning to end) and backward (from end to beginning). Since the list positions for the two matrices must align, the POS probabilities are inserted at the head of the *backward* list as the calculation moves toward the beginning of the sentence. After the passes are completed, the forward and backward probabilities for each POS at each position are multiplied together to estimate a likelihood, and the POS with the maximum likelihood is predicted at each position. The forward-backward implementation of the HMM-VE algorithm was much more accurate than the forward pass by itself; in fact, it was more accurate than the Simplified algorithm. 
    
3. **HMM Viterbi**: Viterbi decoding differs from VE in several ways:
    + Probabilities are only evaluated in the forward direction. 
    + The Markov property is applied to each POS; i.e., joint distributions do not have to be accumulated across the entire matrix. Instead, the probability of any POS--i.e., the dynamic programming values in the matrix--are not the sum of all the previous joint distributions multiplied by transition probabilities. Instead, once the immediately previous POS probabilities are multiplied by transition probabilities, the maximum likelihood from the most likely previous POS is selected for each POS in the current position. These probabilities are then multiplied by emission probabilities to yield the POS probabilities for the current word.
    + The most likely predecessor POS for each POS for the current sentence position is tracked in a separate list of lists. In our code this list is named `backtrack`.

    When the POS probabilities for the final word are calculated, the POS with the maximum likelihood is selected as the prediction for the final POS in the sentence. Its most likely predecessor is fetched from the `backtrack` list and becomes the prediction for the next to last POS in the sentence. This process is followed iteratively until the beginning of the sentence is reached, yielding a predicted POS sequence for the entire sentence.
    
    Our implementation used base-2 log probabilities rather than raw probabilities in order to accelerate math calculations and to avoid underflow errors. To avoid math domain errors when converting raw probabilities of 0.0 to logarithms, we created the following vectorized function:
    
    ```python
    safeLog = np.vectorize(lambda p: log(p, 2) if p > 0 else -10000.0, otypes=[np.float])
    ```
    
    Applying this function to an array of raw probabilities yielded suitable log probabilities. For example:
    
    ```python
    self.initial_state_probs_log = safeLog(self.initial_state_probs)
    ```

### Training method
The `pos_solver.train` method is responsible for calculating the various probabilities required by the three algorithms from the training corpus.

The Simplified algorithm requires the following probabilities:
+ **P(S | W)**. This is implemented as a dictionary of words with each word pointing to a list with 12 elements. Those elements contain the probability of a part-of-speech based on the training data. 
+ **Raw P(S)** for words not seen in the training data. This enables identification of the most common part-of-speech, which in our training corpus was the noun.
 
The VE algorithm requires the following probabilities: 
+ **P(S) for initial state**. 
+ **P(S) for end state**.
+ **P(W | S)** -- i.e., emission probabilities. 
+ **P(S | S<sub>previous</sub>)** -- i.e., transition probabilities.

The initial and end state probabilities are both calculated similarly; any time a part-of-speech is at the front or end of a sentence,  the count for that part-of-speech is incremented by 1. Likewise, for each word encountered, we add it to a dictionary of words for each part-of-speech and to our lexicon if it is not already present. We then increment the count by one. Finally, for all words beyond the first, we record the last and current part-of-speech tag and then increment by the appropriate POS transition count. Once training is completed, each probability variable is divided by the total observations in order to normalize.
 
The Viterbi algorithm requires the same probabilities as HMM-VE. However, probabilities are expressed in logarithmic rather than decimal form, as discussed above.

### Evaluation of Posterior Probability

Given the probabilities calculated during training, the posterior probability of any POS sequence for a sentence -- even the ground truth -- can be calculated using Bayes Law as follows:

<em>P(S<sub>1</sub>,...S<sub>n</sub> | W<sub>1</sub>...W<sub>n</sub>) = 
    P(W<sub>1</sub> | S<sub>1</sub>)...P(W<sub>n</sub> | S<sub>n</sub>)P(S<sub>n</sub> | S<sub>n-1</sub>)...P(S<sub>2</sub> | S<sub>1</sub>) / P(W<sub>1</sub>)P(W<sub>2</sub>)...P(W<sub>n</sub>)</em>

Our implementation adds log probabilities in lieu of multiplying decimal probabilities for the reasons discussed above. When a word W<sub>k</sub> in the test set is absent from training, an extremely low log probability is assigned to P(W<sub>k</sub> | S) in order to avoid math domain errors. The denominator is excluded from the calculation since it does not add any information to the comparison of the posteriors for the algorithms and ground truth.

### Results

Since Viterbi decoding can be mathematically shown to find the most likely POS sequence in the Hidden Markov Model, it is not surprising that the Viterbi posterior was the highest for all 2000 sentences. We verified this outcome by adding the following code to `pos_scorer.print_results()`:

```python
if posteriors[keys[3]] < max(posteriors.values()):
    print ("PROBLEM! Viterbi did not find the max likelihood sequence!")
```

Since the "PROBLEM" message was never printed to console, we were able to conclude that the Viterbi posterior was always greater than or equal to all other posteriors. VE's posterior frequently tied Viterbi's, but was occasionally lower. Curiously, the ground truth posterior was never higher than the Viterbi posterior, and frequently lower by a very substantial amount. For Hidden Markov Models, ground truth is often stranger than fiction.

Given that it predicts the greatest posterior probability, it is not surprising that the Viterbi decoder is the most accurate predictor of POS sequences. It is followed closely by the VE algorithm, as seen below: 

Algorithm | Word Accuracy | Sentence Accuracy
------ | -------- | --------
Simplified | 93.95% | 47.50%
Variable Elimination | 95.08% | 54.30%
Viterbi | 95.05% | 54.45%

The Simplified algorithm is surprisingly accurate, lagging only a little behind the other algorithms. This indicates that most of the POS information is contained in the tokens themselves; transition probabilities and initial/end state probabilities do not contribute much extra predictive power. 

It is a bit surprising that VE's word accuracy is ever so slightly superior to Viterbi's. The difference is so tiny that this anomaly is likely due to a few random results in the test set.



