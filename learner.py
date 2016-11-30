#!/usr/bin/env python
import optparse, sys, os, bleu, random, math, string
from collections import namedtuple

#faster sampleing
def samples(nbest, alpha, tau):
    random.seed()
    for _ in xrange(tau):
        s1 = random.choice(nbest)
        s2 = random.choice(nbest)
        if (s1 != s2) and (math.fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha):
            yield (s1, s2) if s1.smoothed_bleu > s2.smoothed_bleu else (s2, s1)

#compute bleu score
def get_nbest(nbest, source, target):
    src = [line.strip().split() for line in open(source).readlines()]
    ref = [line.strip().split() for line in open(target).readlines()]
    translations = [line.strip().split("|||") for line in open(nbest).readlines()]
    nbests = [[] for _ in ref]
    original_feature_count = 0
    sys.stderr.write("Computing smoothed bleu...")
    translation = namedtuple("translation", "features, smoothed_bleu")
    for (i, sentence, features) in translations:
        (i, sentence, features) = (int(i), sentence.strip(), [float(f) for f in features.strip().split()])
        sentence_split = sentence.strip().split()
        stats = tuple(bleu.bleu_stats(sentence_split, ref[i]))
        nbests[i].append(translation(features, bleu.smoothed_bleu(stats)))
    return nbests

#perceptron
def computePRO(nbests, tau, eta, xi, alpha, epochs):
    theta = [random.uniform(-1.0*sys.maxint, 1.0) for _ in xrange(len(nbests[0][0].features))]
    for epoch in xrange(epochs):
        mistakes = 0.0
        num = 0.0
        sys.stderr.write("Iteration %s..." % (epoch+1))
        for nbest in nbests:
            for (s1, s2) in sorted(samples(nbest, alpha, tau), key=lambda (s1, s2): s2.smoothed_bleu - s1.smoothed_bleu)[:xi]:
                if sum([x * y for x, y in zip(theta, s1.features)]) <= sum([x * y for x, y in zip(theta, s2.features)]):
                    mistakes += 1
                    # theta += eta * (s1.features - s2.features)
                    # update = eta * (s1.features - s2.features)
                    update = [item * eta for item in [x - y for x, y in zip(s1.features, s2.features)]]
                    # theta = theta + update
                    theta = [x + y for x, y in zip(theta, update)]# theta += eta * (s1.features - s2.features)
                num += 1
        sys.stderr.write("\n")
        sys.stderr.write("Mistakes --> %s...\n" % str(int(mistakes)))
        theta = [t/(num) for t in theta]
    sys.stderr.write("\n")
    return theta

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tau", dest="tau", default=5000, type="int", help="Samples generated from n-best list per input sentence (default=5000)")
    optparser.add_option("-a", "--alpha", dest="alpha", default=0.21, type="float", help="Sampler acceptance cutoff (default=0.21)")
    optparser.add_option("-x", "--xi", dest="xi", default=100, type="int", help="Training data generated from the samples tau (default=100)")
    optparser.add_option("-r", "--eta", dest="eta", default=0.1, type="float", help="Perceptron learning rate (default=0.1)")
    optparser.add_option("-i", "--epochs", dest="epochs", default=5, type="int", help="Number of epochs for perceptron training (default=5)")
    optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
    optparser.add_option("-e", "--target", dest="target", default=os.path.join("data", "train.en"), help="Target file")
    optparser.add_option("-f", "--source", dest="source", default=os.path.join("data", "train.fr"), help="Source file")
    (opts, _) = optparser.parse_args()
    nbests = get_nbest(opts.nbest, opts.source, opts.target)
    weights = computePRO(nbests, opts.tau, opts.eta, opts.xi, opts.alpha, opts.epochs)
    print "\n".join([str(weight) for weight in weights])
