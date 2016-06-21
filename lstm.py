#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import glob
import json
import argparse
import numpy as np

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

from fakespeare.play import format_play

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output-dir", default="output",
                    help="the output directory")
args = parser.parse_args()
outdir = args.output_dir

text = []
for fn in glob.glob("plays/*.txt"):
    with open(fn, "r") as f:
        text.append(format_play(f.read()))
text = "\n".join(text)

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 50
step = 3
batch_size = 128
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# maxdata = len(X) // batch_size * batch_size
# X = X[:maxdata]
# y = y[:maxdata]


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True,
               input_shape=(maxlen, len(chars))))
               # stateful=True,
               # batch_input_shape=(batch_size, maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False, ))
               # stateful=True,
               # batch_input_shape=(batch_size, maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# Save the model.
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, 'architecture.json'), 'w') as f:
    f.write(model.to_json())
with open(os.path.join(outdir, 'maps.json'), 'w') as f:
    json.dump(dict(char_indices=char_indices, indices_char=indices_char,
                   maxlen=maxlen, step=step, batch_size=batch_size), f)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=batch_size, nb_epoch=1)

    model.save_weights(os.path.join(outdir,
                                    'weights_{0:05d}.h5'.format(iteration)))

    # model.reset_states()
