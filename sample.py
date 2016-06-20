#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import glob
import json
import random
import argparse
import numpy as np

from keras.models import model_from_json

parser = argparse.ArgumentParser()
parser.add_argument("arch", help="the architecture JSON")
parser.add_argument("maps", help="the mappers JSON")
parser.add_argument("weights", help="the weights JSON")
args = parser.parse_args()

with open(args.arch, "r") as f:
    model = model_from_json(f.read())
model.load_weights(args.weights)
with open(args.maps, "r") as f:
    maps = json.load(f)
    char_indices = maps["char_indices"]
    indices_char = dict((int(k), v) for k, v in maps["indices_char"].items())
    maxlen = maps["maxlen"]


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

sentence = "\n\n**CHAR0**\n\nDark was the night as sometimes whatever"[:maxlen]
print(len(sentence))
generated = "" + sentence
for i in range(100):
    x = np.zeros((1, maxlen, len(char_indices)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 1.0)
    next_char = indices_char[next_index]
    generated += next_char
    sentence = sentence[1:] + next_char

print(generated)

# # train the model, output generated text after each iteration
# for iteration in range(1, 60):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
#     model.fit(X, y, batch_size=128, nb_epoch=1)

#     model.save_weights(os.path.join(outdir,
#                                     'weights_{0:05d}.h5'.format(iteration)))

#     start_index = random.randint(0, len(text) - maxlen - 1)

#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print()
#         print('----- diversity:', diversity)

#         generated = ''
#         sentence = text[start_index: start_index + maxlen]
#         generated += sentence
#         print('----- Generating with seed: "' + sentence + '"')
#         sys.stdout.write(generated)

#         for i in range(400):
#             x = np.zeros((1, maxlen, len(chars)))
#             for t, char in enumerate(sentence):
#                 x[0, t, char_indices[char]] = 1.

#             preds = model.predict(x, verbose=0)[0]
#             next_index = sample(preds, diversity)
#             next_char = indices_char[next_index]

#             generated += next_char
#             sentence = sentence[1:] + next_char

#             sys.stdout.write(next_char)
#             sys.stdout.flush()
#         print()
