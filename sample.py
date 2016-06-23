#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import sys
import json
import argparse
import numpy as np

from keras.models import model_from_json

parser = argparse.ArgumentParser()
parser.add_argument("arch", help="the architecture JSON")
parser.add_argument("maps", help="the mappers JSON")
parser.add_argument("weights", help="the weights JSON")
parser.add_argument("-o", "--output", default=None,
                    help="the output sample file")
parser.add_argument("-n", "--nchars", default=None, type=int,
                    help="the total number of characters to generate")
parser.add_argument("-t", "--temp", default=0.5, type=float,
                    help="the temperature of the sample")
args = parser.parse_args()

with open(args.arch, "r") as f:
    model = model_from_json(f.read())
model.load_weights(args.weights)
with open(args.maps, "r") as f:
    maps = json.load(f)
    char_indices = maps["char_indices"]
    indices_char = dict((int(k), v) for k, v in maps["indices_char"].items())
    maxlen = maps["maxlen"]
    batch_size = maps["batch_size"]


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

random_seed = input("Random seed: ")
if len(random_seed):
    np.random.seed(int(random_seed))
seed = input("Seed dialog: ")

sentence = "<scene>\n\n_Enter CHAR10_\n\n**CHAR10**\n\n" + seed
generated = "" + sentence
sentence = sentence[-maxlen:]
if len(sentence) < maxlen:
    sentence = ("\n" * (maxlen - len(sentence))) + sentence
sys.stdout.write(generated)
sys.stdout.flush()

if args.output is not None:
    with open(args.output, "w") as f:
        f.write(generated)

i = 0
while True:
    x = np.zeros((1, maxlen, len(char_indices)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, args.temp)
    next_char = indices_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

    if args.output is not None:
        with open(args.output, "a") as f:
            f.write(next_char)

    i += 1
    if args.nchars is not None and i >= args.nchars:
        break
