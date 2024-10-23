#!/usr/bin/env python3.9


from re import sub
import random

def text_preprocess(sentence):

    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = sub('[(,!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence

def text_enhance(sentence):
    sentences = sentence.split("seconds")[:-1]
    random.shuffle(sentences)
    # concatenate the sentences
    text_shuffle = ""
    for s in sentences:
        text_shuffle += s.strip() + " seconds "
    return text_shuffle