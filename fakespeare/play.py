#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import re

__all__ = []


def find_characters(txt):
    groups = re.findall(r"\*\*(.+)\*\*", txt)
    return dict([(n, "CHAR{0}".format(i))
                 for i, n in enumerate(sorted(set(groups)))])


def format_play(txt):
    chars = find_characters(txt)
    for k, v in chars.items():
        txt = txt.replace("**"+k+"**", "**"+v+"**")
    for k, v in chars.items():
        txt = txt.replace(k, v)
    return txt


if __name__ == "__main__":
    txt = open("plays/Antony_and_Cleopatra.txt", "r").read()
    format_play(txt)
