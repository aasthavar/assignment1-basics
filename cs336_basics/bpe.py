import regex as re
from configs import config
# from pretokenization_example import find_chunk_boundaries

config = config["tokenizer"]
config["PAT"] = re.compile(config["PAT"])

# unoptimized: loops over all keys
def update_freq_table(
    freq_table: dict[tuple[bytes], int], 
    max_pair: tuple[bytes, bytes]
) -> dict[tuple[bytes], int]:
    """
    return a new frequency table after merging `pair` everywhere.
    """
    a, b = max_pair
    merged = a + b
    new_freq_table = {}
    for key, value in freq_table.items():
        L = len(key)
        i=0
        out = []
        while i < L:
            if (i+1 < L) and (key[i] == a) and (key[i + 1] == b):
                out.append(merged)
                i+=2
            else:
                out.append(key[i])
                i+=1
        new_key = tuple(out)
        new_freq_table[new_key] = new_freq_table.get(new_key, 0) + value
    return new_freq_table

def get_max_bytes_pair(
    bytes_pair_counts: dict[tuple[bytes, bytes], int]
) -> tuple[bytes, bytes]:
    """
    returns pair with max count. If counts tie, pick lexicographically greatest pair.
    """
    # tuple comparison: first by count, then by pair lexicographically
    return max(
        bytes_pair_counts.items(),
        key=lambda item: (item[1], item[0])
    )[0]

def get_bytes_pair_counts(
    freq_table: dict[tuple[bytes], int]
) -> dict[tuple[bytes, bytes], int]:
    """build a dictionary of byte-pair frequencies"""
    bytes_pair_counts: dict[tuple[bytes, bytes], int] = {}
    for key, value in freq_table.items():
        key_len = len(key)
        for i in range(key_len-1):
            pair = (key[i], key[i+1])
            bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + value
    return bytes_pair_counts
    
def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # step 1: init vocab
    vocab: dict[int, bytes] = {}
    vid = len(vocab)
    for token in special_tokens:
        vocab[vid] = token.encode("utf-8")
        vid += 1
    for i in range(256):
        vocab[vid] = bytes([i])
        vid += 1
    # print(f"vocab: {vocab}")
    
    # step 2: pre-tokenize
    freq_table: dict[tuple[bytes], int] = {}
    with open(input_path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
        escaped = [re.escape(token) for token in special_tokens]
        documents = re.split("|".join(escaped), text)
        for doc in documents:
            words = config["PAT"].findall(doc)
            for word in words:
               key = tuple([bytes([x]) for x in word.encode("utf-8")])
               freq_table[key] = freq_table.get(key, 0) + 1
    # print(f"total unique words: {len(freq_table)}")
    
    # step 3: compute merges
    merges: list[tuple(bytes, bytes)] = []
    while vid < vocab_size:
        # step 3.1: count byte-pair frequencies
        bytes_pair_counts = get_bytes_pair_counts(freq_table)
        if not bytes_pair_counts: # no merge byte pairs to merge
            break
        
        # step 3.2: find the pair with max count. to break ties, choose lexicographically higher pair
        pair = get_max_bytes_pair(bytes_pair_counts)

        # step 3.3: update freq_table (by merging the bytes pair), vocab, merges, 
        freq_table = update_freq_table(freq_table, pair)
        merges.append(pair)
        vocab[vid] = pair[0]+pair[1]
        vid += 1
    return vocab, merges

if __name__ == "__main__":
    print(f"training on data from: {config["input_path"].split("/")[-1]}")
    vocab, merges = train_bpe(
        input_path=config["input_path"],
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"]
    )
    # print(f"merges: {merges}")
    # print("\nvocab:")
    # for i in range(256, len(vocab)):
    #     print(f"{i}, {vocab[i]}")
    
    # vocab = {"a": "b"}
    # vocab |= ({"b": "c"})
    # print(vocab)
    