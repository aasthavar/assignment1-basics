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
Find code in bpe.py


### Problem (train_bpe_tinystories)
(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size
of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary.
Serialize the resulting vocabulary and merges to disk for further inspection. How many hours
and memory did training take? What is the longest token in the vocabulary? Does it make sense?
Ans: 
- time took to train:  78.9581s + 81.2624s = 2.7 mins. Note: the time is slightly higher than 2 mins because of the tqdm stuff. without it its under 2 mins.
- didn't capture the memory.
- longest token in vocab: "Ġaccomplishment" which means " accomplishment". Ġ is a marker for a leading space.

(b) Profile your code. What part of the tokenizer training process takes the most time?
Ans: 
- profile:
    [TIMER] init vocab: 0.0000s
    [TIMER] pre-tokenization: 78.9581s
    [TIMER] compute merges: calc counts: 50.0580s
    [TIMER] compute merges: bpe merge: 100%|█████████████████████████████████████████████████████████████████████████████| 9743/9743 [00:31<00:00, 312.30merge/s]
    [TIMER] compute merges (total): 81.2624s
- In this case pre-tokenization took slightly more time.


### Problem (train_bpe_expts_owt)
(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
Ans: 
- For the valid dataset: longest token is: "----------------------------------------------------------------". Nope it doesn't make sense. 
- 

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
- Thoughts:
  - TinyStories -> clean, curated prose -> long words dominate
  - OpenWebText -> raw web text -> formatting artifacts dominate
- Takeway: BPE optimizes for frequency, not semantics. BPE tokenizers mirror the statistics of their training data—clean data yields semantic tokens, noisy data yields structural artifacts.


### Problem (tokenizer)
See code in tokenizer.py


### Problem (tokenizer_experiments)
(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
Ans: 
- TinyStories tokenizer average compression: 4.11 bytes/token.
- didn't check for owt


(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
Ans: Compression degrades significantly: bytes/token increases because the TinyStories tokenizer lacks merges for web-specific artifacts (punctuation runs, markup, rare words). Qualitatively, text is split into many smaller byte-level tokens, leading to longer token sequences.

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
Ans: 
- for tinystoreis, throughput: 765388.5096985975 bytes/sec
- time = 1.08e6 seconds ~300 hours ~12.5 days on a single CPU core

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
Ans: uint16 can represent up to 65,536 token IDs, which safely covers vocab sizes like 10K–32K. It halves memory usage compared to int32, improving storage efficiency and cache performance without loss of information.