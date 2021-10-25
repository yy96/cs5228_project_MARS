import string

def _remove_punctuation(word):
    punch = string.punctuation
    for p in punch:
        word = word.replace(p, '')
    word = word.replace('…', '')
    word = word.replace('–', '')
    word = word.replace("’", '')
    return word