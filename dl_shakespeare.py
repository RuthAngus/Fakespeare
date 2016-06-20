#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import requests
from bs4 import BeautifulSoup
from html2text import html2text

__all__ = ["dl_scripts"]


BASE_URL = "http://shakespeare.mit.edu"


def dl_scripts():
    url = BASE_URL
    r = requests.get(url)
    tree = BeautifulSoup(r.text, "html.parser")
    os.makedirs("plays", exist_ok=True)
    for a in tree.find_all("a")[2:-7]:
        link = a.get("href").split("/")[0]
        title = a.text.strip().replace(" ", "_")
        title = title.replace("\n", "_")
        fn = "plays/" + title + ".txt"

        r = requests.get(BASE_URL + "/" + link + "/full.html")
        body = html2text(r.text.replace("blockquote", "p"))
        body = body[body.index("### ACT I"):]
        with open(fn, "w") as f:
            f.write(body)


if __name__ == "__main__":
    dl_scripts()
