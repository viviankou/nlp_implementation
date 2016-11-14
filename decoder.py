#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
import copy
from math import log10

def score(cp, f):
  lm_prob = 0.0
  lm_state = lm.begin()
  num_f_translated = 0
  for n, (ep, fp) in enumerate(cp):
    if ep != None and fp != None:
      num_f_translated += len(fp)
      for word in ep.english.split():
        (lm_state, word_logprob) = lm.score(lm_state, word)
        lm_prob += word_logprob
      lm_prob += lm.end(lm_state) if num_f_translated == len(f) else 0.0
  tm_prob = 0.0
  for (ep, fp) in cp:
    if ep != None:
      tm_prob += ep.logprob

  return (lm_prob + tm_prob)

def swap(cp):
  swaps = []
  for i in xrange(len(cp)-1):
    for j in xrange(i, len(cp)):
      swapped = copy.deepcopy(cp)
      temp = swapped[i]
      swapped[i] = swapped[j]
      swapped[j] = temp
      swaps.append(swapped)
  return swaps

def replace(cp):
  replaces = []
  for n, p in enumerate(cp):
    if p[1] in tm:
      ts = tm[p[1]]
      for t in ts:
        if p[0] != t:
          replaced = copy.deepcopy(cp)
          replaced[n] = (t, p[1])
          replaces.append(replaced)
  return replaces

def merge(cp):
  merges = []
  for i in xrange(1, len(cp)-1):
    f1 = cp[i][1]
    f2 = cp[i+1][1]
    if f1 and f2 and (f1 + f2) in tm:
      for t in tm[f1+f2]:
        merged = copy.deepcopy(cp)
        merged.remove(cp[i+1])
        merged[i] = (t, f1+f2)
        merges.append(merged)
  if len(cp) >= 3:
    for i in xrange(1, len(cp)-2):
      f1 = cp[i][1]
      f2 = cp[i+1][1]
      f3 = cp[i+2][1]
      if f1 and f2 and f3 and (f1 + f2 + f3) in tm:
        for t in tm[f1+f2+f3]:
          merged = copy.deepcopy(cp)
          merged.remove(cp[i+1])
          merged.remove(cp[i+2])
          merged[i] = (t, f1+f2+f3)
          merges.append(merged)
  return merges

def split(cp):
  splits = []
  for n, i in enumerate(cp):
    french_phrase = cp[n][1]
    if french_phrase != None:
      if len(french_phrase) > 1:
        for j in xrange(1, len(french_phrase)):
          s1 = french_phrase[0:j]
          s2 = french_phrase[j:]
          if s1 in tm and s2 in tm:
            for ts1 in tm[s1]:
              for ts2 in tm[s2]:
                spl = copy.deepcopy(cp)
                spl[n] = (ts1, s1)
                spl.insert(n+1, (ts2, s2))
                splits.append(spl)
  return splits

def greedy_decode(f, seed):
  iters = 100
  current = seed
  for i in xrange(iters):
    s_current = score(current, f)
    s = s_current
    n = swap(current) + merge(current) + replace(current) + split(current)
    for h in n:
      c = score(h, f)
      if c > s:
        s = c
        best = h
    if s == s_current:
      return current
    else:
      current = best
  return current

def find_beam(stack):
  beam = sorted(stack.itervalues(), key=lambda h: -h.logprob)
  return beam

def beam_decode(f):
  opt_eta = opts.eta
  opt_s = opts.s
  opt_distort = opts.distort
  opt_k = opts.k
  if len(f) < 9:
      opt_eta = 0.5
      opt_s = 200
      opt_distort = 3
  if len(f) > 15:
      opt_eta = 0.9
      opt_s = 500
      opt_distort = 3
      opt_k = 20

  bit_vec = [0] * len(f)

  # Initialising the hypothesis
  # logprob = previous state logprob + phrase logprob, lm_state = last 2 words of the phrase,
  # last_ind = index of the last word in last phrase in previous state, predecessor = previous state
  hypothesis = namedtuple("hypothesis", "logprob, lm_state, bit_vec, last_ind, predecessor, phrase, fphrase")
  initial_hypothesis = hypothesis(0.0, lm.begin(), bit_vec, 0, None, None, None)

  # Creating the stacks
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis

  # Iterating over all stacks except the one where all words
  # are translated
  for i, stack in enumerate(stacks[:-1]):

      # Find the sorted and pruned stack
      # apply beam limit
      beam = find_beam(stack)

      # Iterating on Pruned Hypotheses - Histogram Pruning
      # beam[:opt_s] = take top opt_s states per stack
      # instead apply beam width limit here e.g limit on number of states to consider per stack
      for h in beam[:opt_s]:

          # Iterating over all phrase possibilities
          probable_phrases = []
          prob_dist_phrases = []

          # ph_range consists of all valid phrases that can follow the h/hypothesis/state
          # x = starting french/source index of the english phrase
          # y = ending french/source index of the english phrase
          ph_range = namedtuple("ph_range", "x, y")
          for x in xrange(0, len(f)):
              for y in xrange(x + 1, len(f) + 1):
                  if 1 in h.bit_vec[x:y]:
                      continue
                  # checking if consecutive phrases are close to each other using distortion limit of 9
                  if abs(h.last_ind + 1 - x) > 9:
                      prob_dist_phrases.append(ph_range(x, y))
                  else:
                      probable_phrases.append(ph_range(x, y))

          if len(probable_phrases) == 0:
              probable_phrases = prob_dist_phrases[:]

          for phrase_range in probable_phrases:

              f_phrase = f[phrase_range.x:phrase_range.y]
              if f_phrase in tm:

                  # Deep copying the bit vector
                  new_bit_vec = h.bit_vec[:]
                  for bt in xrange(phrase_range.x, phrase_range.y):
                      new_bit_vec[bt] = 1

                  for phrase in tm[f_phrase][:opt_k]:

                      # Adding the phrase translation probability
                      logprob = h.logprob + phrase.logprob
                      lm_state = h.lm_state

                      # Computing the language probability for english phrase
                      for word in phrase.english.split():
                          (lm_state, word_logprob) = lm.score(lm_state, word)
                          logprob += word_logprob
                      if 0 not in new_bit_vec:
                          logprob += lm.end(lm_state)
                      logprob += log10(opt_eta) * abs(h.last_ind + 1 - phrase_range.x)

                      # Check for the correct stack number
                      # new_bit_vec.count(1) returns how many time 1 occurs in new_bit vector (e.g number of translated words), that number will be the stack number
                      if i + phrase_range.y - phrase_range.x != new_bit_vec.count(1):
                          sys.stderr.write("Stack Error")
                      # Create the new hypothesis
                      new_hypothesis = hypothesis(logprob, lm_state, new_bit_vec, phrase_range.y - 1, h, phrase, f_phrase)

                      # Recombination in the stack
                      # Add method in algorithm
                      if lm_state not in stacks[i + phrase_range.y - phrase_range.x] or \
                                      stacks[i + phrase_range.y - phrase_range.x][
                                          lm_state].logprob < logprob:  # second case is recombination
                          stacks[i + phrase_range.y - phrase_range.x][lm_state] = new_hypothesis

  winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  return hyp_to_phrases(winner)

def hyp_to_phrases(hyp):
  phrases = []
  def get_phrases(hyp, ps):
    if hyp == None:
      return
    else:
      ps.insert(0, (hyp.phrase, hyp.fphrase))
      get_phrases(hyp.predecessor, ps)
  get_phrases(hyp, phrases)
  return phrases

def get_trans_options(h, f):
  options = []
  for fi in xrange(len(f)):
    for fj in xrange(fi+1, len(f)+1):
      # check if the range is unmarked
      unmarked = all(lambda x: h.marked[x]==0 for m in range(fi, fj))
      if unmarked:
        if f[fi:fj] in tm:
          phrases = tm[f[fi:fj]]
          for p in phrases:
            options.append((p, (fi, fj)))
  return options

def print_phrases(phrases):
  s = ""
  for p in phrases:
    if p[0] != None:
      s += p[0].english + " "
  print s

if __name__ == "__main__":
  optparser = optparse.OptionParser()
  optparser.add_option("-i", "--input", dest="input", default="data/input",
                       help="File containing sentences to translate (default=data/input)")
  optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm",
                       help="File containing translation model (default=data/tm)")
  optparser.add_option("-l", "--language-model", dest="lm", default="data/lm",
                       help="File containing ARPA-format language model (default=data/lm)")
  optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                       help="Number of sentences to decode (default=no limit)")
  optparser.add_option("-k", "--translations-per-phrase", dest="k", default=10, type="int",
                       help="Limit on number of translations to consider per phrase (default=1)")
  optparser.add_option("-s", "--stack-size", dest="s", default=500, type="int", help="Maximum stack size (default=1)")
  optparser.add_option("-e", "--eta", dest="eta", default=0.6, type="float",
                       help="Eta Value for distortion model (default=0.6)")
  optparser.add_option("-d", "--distort", dest="distort", default=4, type="int",
                       help="Maximum distortion length (default=4)")
  optparser.add_option("-a", "--alpha", dest="alpha", default=0.0001, type="float",
                       help="Alpha value for threshold pruning (default=0.0001)")
  optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                       help="Verbose mode (default=off)")
  opts = optparser.parse_args()[0]

  tm = models.TM(opts.tm, opts.k)
  lm = models.LM(opts.lm)
  french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

  # tm should translate unknown words as-is with probability 1
  for word in set(sum(french,())):
    if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, 0.0)]

  sys.stderr.write("Decoding %s...\n" % (opts.input,))
  for f in french:
    # The following code implements a monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of 
    # the first i words of the input sentence. You should generalize
    # this so that they can represent translations of *any* i words.
    seed = beam_decode(f)
    decoded = greedy_decode(f, seed)
    print_phrases(decoded)
  
