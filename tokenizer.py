import torch
from torch import nn
import math
import random

vocab_size = 41

emedding_dimensions = 32

class Tokenizer:
  def __init__(self):
    self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?")


    self.charnum = {char: i for i, char in enumerate(self.chars)}
    self.numchar = {i: char for i, char in enumerate(self.chars)}

  def tokenize(self, string):

    return [self.charnum.get(char, self.charnum[' ']) for char in string.lower()]

  def decode(self, token_ids):
    return "".join([self.numchar.get(item, ' ') for item in token_ids])
