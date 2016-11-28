#!/usr/bin/env python
import optparse
import sys
import os
import bleu
import random
import math
import time
import itertools
from collections import namedtuple

# samples generated from n-best list per input sentence
tau = 5000
# sampler acceptance cutoff
alpha = 0.1
# training data generated from the samples tau
xi = 100
# perceptron learning rate
eta = 0.1
# number of epochs for perceptron training
epochs = 5
# lines = [index for index,line in enumerate(open(opts.reference))]
# nbests = [[] for x in xrange(lines[-1]+1)]
# Number of sentences are 1989
nbests = [[] for x in xrange(1989)]
# It saves the already seen tuples
Cache = [[] for x in xrange(1989)]


def computeBleu(system, reference):
    stats = [0 for i in xrange(10)]
    stats = [sum(scores) for scores in
             zip(stats, bleu.bleu_stats(system, reference))]
    return bleu.smoothed_bleu(stats)


def Check_Tuple(e1, e2, c):
    if tuple([e1, e2]) in c:
        # sys.stderr.write("1")
        return False
    elif tuple([e2, e1]) in c:
        # sys.stderr.write("1")
        return False
    else:
        return True


def get_sample(nbest, cache):
  # sample contains a list of tuples for a source sentence, where each tuple = (s1,s2)
    sample = []
    if len(nbest) >= 2:
        for i in range(tau):
            # select returns a list of 2 randomly selected items from nbest
            select = random.sample(nbest, 2)
            s1 = select[0]
            s2 = select[1]
            while Check_Tuple(s1, s2, cache) is False:
                select = random.sample(nbest, 2)
                s1 = select[0]
                s2 = select[1]

            if math.fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha:
                if s1.smoothed_bleu > s2.smoothed_bleu:
                    # sample += (s1, s2)
                    sample.append((s1, s2))
                else:
                    # sample += (s2, s1)
                    sample.append((s2, s1))
            else:
                continue
    else:
        s1 = nbest[0]
        s2 = nbest[0]
        sample.append((s1, s2))
    return sample

def count_possible_untranslated(src, sentence):
    filtered = itertools.ifilter(lambda h: not any(c.isdigit() for c in h) , sentence) # strip numerals and words containing them
    possible_untranslated = -1.0 * (len(set(src).intersection(filtered)) + 1) # add 1 as -1 is our best score
    return possible_untranslated

def computeNBests():
    src = [line.strip().split() for line in open(opts.source).readlines()]
    ref = {int(index): line for index, line in enumerate(open(opts.reference))}
    for line in open(opts.nbest):
        # ith french sentence, english translated sentence, feature weights
        (i, sentence, features) = line.strip().split("|||")
        system = sentence.strip().split()
        reference = ref[int(i.strip())].strip().split()
        score = computeBleu(system, reference)
        features = [float(h) for h in features.strip().split()]
        sentence_split = sentence.strip().split()
        #stats = tuple(bleu.bleu_stats(sentence_split, ref[i]))
        features.append(count_possible_untranslated(src[int(i)], sentence_split))
        #score = sum(features)
        translation = namedtuple("translation",
                                 "sentence, smoothed_bleu, features")
        nbests[int(i)].append(translation(sentence, score, features))

        # theta = [float(1.0) for _ in xrange(len(features))]
        theta = [random.uniform(-1.0*sys.maxint, 1.0)
                 for _ in xrange(len(features))]
    return [nbests, theta]


def computePRO(nbests, theta):
    for i in range(epochs):
        mistakes = 0
        # nbest contains a list of (english translation, bleu score, features) tuples for a source sentence
        for i, nbest in enumerate(nbests):
            sample = get_sample(nbest, Cache[i])
            Cache[i] = Cache[i] + sample
            # sort the tau samples from get_sample() using s1.smoothed_bleu - s2.smoothed_bleu
            sample.sort(key=lambda tup: math.fabs(tup[0].smoothed_bleu - tup[1].smoothed_bleu))
            sample.reverse()
            # keep the top xi (s1, s2) values from the sorted list of samples
            sample = sample[:xi]
            # do a perceptron update of the parameters theta:
            for tup in sample:
                s1, s2 = tup
                # if theta * s1.features <= theta * s2.features:
                if sum([x * y for x, y in zip(theta, s1.features)]) <= sum([x * y for x, y in zip(theta, s2.features)]):
                    mistakes += 1
                    # theta += eta * (s1.features - s2.features)  # this is vector addition!
                    # update = eta * (s1.features - s2.features)
                    update = [item * eta for item in [x - y for x, y in zip(s1.features, s2.features)]]
                    # theta = theta + update
                    theta = [x + y for x, y in zip(theta, update)]
                    # sys.stderr.write(" ".join([str(weight) for weight in theta]))
                    # sys.stderr.write("\n")
        sys.stderr.write("Mistakes --> %s...\n" % str(int(mistakes)))
    return theta


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-f", "--source", dest="source", default=os.path.join("data", "train.fr"), help="Source file")
    optparser.add_option("-n", "--nbest",
                         dest="nbest",
                         default=os.path.join("data", "train.nbest"),
                         help="N-best file")
    optparser.add_option("-r", "--reference",
                         dest="reference",
                         default=os.path.join("data", "train.en"),
                         help="English reference sentences")
    (opts, _) = optparser.parse_args()

    start = time.time()
    output = computeNBests()
    end = time.time()
    sys.stderr.write("Compute NBest is Finished in %s seconds!\n" % str(end-start))

    start = time.time()
    weights = computePRO(output[0], output[1])
    end = time.time()
    sys.stderr.write("Compute PRO is Finished in %s seconds!\n" % str(end-start))
    print "\n".join([str(weight) for weight in weights])