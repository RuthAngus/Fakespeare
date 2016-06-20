#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import re
from collections import defaultdict

__all__ = ["find_characters", "format_play"]


def find_characters(txt):
    groups = re.findall(r"\*\*(.+)\*\*", txt)
    chars = defaultdict(lambda: "CHAR{0}".format(len(chars)))
    [chars[k] for k in groups]
    return chars


def format_play(txt):
    chars = find_characters(txt)
    for k, v in chars.items():
        txt = txt.replace("**"+k+"**", "**"+v+"**")
    for k, v in chars.items():
        txt = txt.replace(k, v)
    txt = "\n".join(filter(lambda x: not x.startswith("#"), txt.splitlines()))
    return txt.strip()


if __name__ == "__main__":
    txt = open("plays/Antony_and_Cleopatra.txt", "r").read()
    print(format_play(txt)[:1000])
