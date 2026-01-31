
# a = 'abc'
# print(a.__repr__())
# print(a)

# ---------------------------- #
# test_string = "hello! こんにちは!"

# utf8_encoded = test_string.encode("utf-8")
# print(len(list(utf8_encoded)))

# utf16_encoded = test_string.encode("utf-16")
# print(len(list(utf16_encoded)))

# utf32_encoded = test_string.encode("utf-32")
# print(len(list(utf32_encoded)))


# ---------------------------- #
# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# print(decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8")))


# ---------------------------- #
# import regex as re

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# print(re.findall(PAT, "some text that i'll pre-tokenize"))
# # ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']

# print(re.findall(PAT, "life is full of surprises! i've many dreams!"))
# # ['life', ' is', ' full', ' of', ' surprises', '!', ' i', "'ve", ' many', ' dreams', '!']

# obj = re.finditer(PAT, "life is full of surprises! i've many dreams!")
# matches = []
# for match in obj:
#     matches.append(match.group())
#     # print(match.group(), match.start(), match.end())
# print(matches)
# # ['life', ' is', ' full', ' of', ' surprises', '!', ' i', "'ve", ' many', ' dreams', '!']



# ---------------------------- #
# # with open("cs336_basics/data/TinyStoriesV2-GPT4-train_vocab.txt") as f:
# with open("cs336_basics/data/owt_valid_vocab.txt") as f:
#     text = f.read()
#     lines = text.split("\n")
#     tokens = []
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         token = line.split(" ")[1]
#         tokens.append(token)

#     print(max(tokens, key=lambda item: len(item)))

# ---------------------------- #
# import regex as re
# special_tokens = ["<|endooftext|>", "<|startoftext|>", "<|toolchoice|>", "aastha!<*&$@"]
# text = """
# <|startoftext|>
# I am fond of oranges.
# Would love to grow a forest of all kinds of trees.
# One day, will build a home near ocean and mountains.
# <|endoftext|>
# """

# escaped = [re.escape(token) for token in special_tokens]
# print(escaped)


# ---------------------------- #
# word = "~aastha~"
# word_encoded = word.encode("utf-8")
# # word_encoded = word.encode("utf-32")
# print(f"len: {len(word_encoded)}")
# print(f"type: {type(word_encoded)}")
# print(f"word_encoded: {word_encoded}")

# for b in word_encoded:
#     print(b, bytes([b]))


# ---------------------------- #
# import torch

# x = torch.ones(5)
# y = torch.zeros(3)

# w = torch.rand(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)

# z = torch.matmul(x, w) + b

# loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# print(loss)


# ---------------------------- #
# import re

# special_tokens = ["*", "!"]
# special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
# pat = "(" + "|".join(map(re.escape, special_tokens_sorted)) + ")" # ['ab', '*', 'c', '*', 'def', '!', '']
# # pat = "|".join(map(re.escape, special_tokens_sorted)) #  ['ab', 'c', 'def', '']

# text = "ab*c*def!"

# parts = re.split(pat, text)
# print(parts)


# ---------------------------- #

# [bs, sl] -> embedding layer -> [bs, sl, d_model]
# [bs, sl, d_in] -> linear layer -> [bs, sl, d_out]

from ntpath import dirname
import torch
import torch.nn as nn
import numpy as np
from fancy_einsum import einsum

din, dout = 2, 3

weights1 = torch.empty((dout, din))
nn.init.normal_(weights1, std=0.02, mean=0.0)
print(f"normal:\n{weights1}")

weights2 = torch.empty((dout, din))
sigma = np.sqrt(2/(din+dout))
nn.init.trunc_normal_(weights2, std=sigma, mean=0.0, a=-3*sigma, b=3*sigma)
print(f"normal truncated:\n{weights2}")

bs, sl = 1, 4
x = torch.randn((bs, sl, din))
print(f"x:\n{x}")

y = einsum("bs sl din, dout din -> bs sl dout", x, weights2)
print(f"y:\n{y}")

# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #