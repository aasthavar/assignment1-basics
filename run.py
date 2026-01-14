
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
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

print(re.findall(PAT, "some text that i'll pre-tokenize"))
# ['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']

print(re.findall(PAT, "life is full of surprises! i've many dreams!"))
# ['life', ' is', ' full', ' of', ' surprises', '!', ' i', "'ve", ' many', ' dreams', '!']

obj = re.finditer(PAT, "life is full of surprises! i've many dreams!")
matches = []
for match in obj:
    matches.append(match.group())
    # print(match.group(), match.start(), match.end())
print(matches)
# ['life', ' is', ' full', ' of', ' surprises', '!', ' i', "'ve", ' many', ' dreams', '!']



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #



# ---------------------------- #