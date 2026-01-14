### Problem (unicode1)

(a) What Unicode character does chr(0) return?
Ans: Returns '\x00'

(b) How does this character’s string representation (__repr__()) differ from its printed representation?
Ans: It's a string and looks exactly like the object defined.

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
>>> chr(0)
'x\00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
'this is a teststring'


### Problem (unicode2)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
Ans: UTF-8 is dominant on the web (98% of all webpages), uses fewer bytes for most text, and has a simple 256-byte vocabulary that’s easy to work with.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'

Ans: Function is decoding each byte independently, but UTF-8 chars can span multiple bytes.
Example string = "hello! こんにちは!"

(c) Give a two byte sequence that does not decode to any Unicode character(s).
Ans: {I don't know}


### Problem (train_bpe)
