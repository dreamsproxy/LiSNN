import os
from glob import glob

raw_texts = []
for p in glob('./datasets/*.txt'):
    with open(p, mode='r', errors='replace', encoding='utf-8') as infile:
        raw_texts.append(infile.readlines())
