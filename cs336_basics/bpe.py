import regex as re
from configs import config
import time
from contextlib import contextmanager
# from pretokenization_example import find_chunk_boundaries

@contextmanager
def timer(name: str, enabled: bool):
    if not enabled:
        yield
        return
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[TIMER] {name}: {end - start:.4f}s")


config = config["tokenizer"]
config["PAT"] = re.compile(config["PAT"])

# # unoptimized version: loops over all token sequences, merge `max_pair` if present
# def apply_bpe_merge(
#     freq_table: dict[tuple[bytes], int],
#     max_pair: tuple[bytes, bytes],
# ) -> dict[tuple[bytes], int]:
#     """
#     return a new frequency table after merging `max_pair` everywhere it occurs.
#     """
#     a, b = max_pair
#     merged = a + b
#     new_freq_table = {}
#     for key, value in freq_table.items():
#         L = len(key)
#         i=0
#         out = []
#         while i < L:
#             if (i+1 < L) and (key[i] == a) and (key[i + 1] == b):
#                 out.append(merged)
#                 i+=2
#             else:
#                 out.append(key[i])
#                 i+=1
#         new_key = tuple(out)
#         new_freq_table[new_key] = new_freq_table.get(new_key, 0) + value
#     return new_freq_table


# # optimized version v1: touch only those keys where you know the `max_pair` exists
# def apply_bpe_merge(
#     freq_table: dict[tuple[bytes], int],
#     max_pair: tuple[bytes, bytes],
#     bytes_pair_to_tokens
# ) -> dict[tuple[bytes], int]:
#     """
#     return a new frequency table after merging `max_pair` everywhere it occurs.
#     """
#     a, b = max_pair
#     merged = a + b
    
#     affected_sequences = bytes_pair_to_tokens[max_pair]
#     if not affected_sequences:
#         return freq_table

#     new_freq_table = freq_table.copy()
    
#     for seq in affected_sequences:
#         count = freq_table[seq]
        
#         i = 0
#         out = []
#         L = len(seq)
#         while i < L:
#             if (i+1 < L) and (seq[i] == a) and (seq[i + 1] == b):
#                 out.append(merged)
#                 i+=2
#             else:
#                 out.append(seq[i])
#                 i+=1
        
#         new_freq_table.pop(seq, None)
        
#         new_key = tuple(out)
#         new_freq_table[new_key] = new_freq_table.get(new_key, 0) + count
    
#     return new_freq_table
    
def apply_bpe_merge(
    freq_table: dict[tuple[bytes, ...], int],
    max_pair: tuple[bytes, bytes],
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    bytes_pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """
    Incrementally update:
      - freq_table
      - bytes_pair_counts
      - bytes_pair_to_tokens

    by merging `max_pair` everywhere it occurs.
    All updates are done in-place.
    """
    a, b = max_pair
    merged = a + b

    affected_sequences = bytes_pair_to_tokens[max_pair]
    if not affected_sequences:
        return

    # copy because we mutate indices during iteration
    affected_sequences = affected_sequences.copy()

    for seq in affected_sequences:
        count = freq_table[seq]

        # ---- build merged sequence ----
        i = 0
        out = []
        L = len(seq)
        while i < L:
            if i + 1 < L and seq[i] == a and seq[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        new_seq = tuple(out)

        # ---- remove old pair contributions ----
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])

            bytes_pair_counts[pair] -= count
            if bytes_pair_counts[pair] == 0:
                del bytes_pair_counts[pair]

            if pair in bytes_pair_to_tokens:
                bytes_pair_to_tokens[pair].discard(seq)
                if not bytes_pair_to_tokens[pair]:
                    del bytes_pair_to_tokens[pair]

        # ---- add new pair contributions ----
        for i in range(len(new_seq) - 1):
            pair = (new_seq[i], new_seq[i + 1])
            bytes_pair_counts[pair] = bytes_pair_counts.get(pair, 0) + count
            bytes_pair_to_tokens.setdefault(pair, set()).add(new_seq)

        # ---- update freq_table ----
        del freq_table[seq]
        freq_table[new_seq] = freq_table.get(new_seq, 0) + count

    # this pair is fully merged and should never appear again
    bytes_pair_counts.pop(max_pair, None)
    bytes_pair_to_tokens.pop(max_pair, None)


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


def build_bytes_pair_to_tokens(
    freq_table: dict[tuple[bytes, ...], int],
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    vocab_size: int
) -> dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]:
    """
    build a mapping from a byte pair to the set of token sequences (tuple[bytes]) 
    in which that byte pair occurs.
    
    only the top `vocab_size` byte pairs (by frequency, with lexicographic 
    tie-breaking) are indexed, since only those pairs are candidates for merging.
    """
    bytes_pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    # sort pairs by (count, pair) descending
    sorted_bytes_pair_counts = sorted(
        bytes_pair_counts.items(),
        key=lambda item: (item[1], item[0]),
        reverse=True,
    )
    for (bytes_pair, _) in sorted_bytes_pair_counts[:vocab_size]:
        tok_seqs_with_bytes_pair = set()
        for token_seq in freq_table.keys():
            # check whether this token sequence contains the bytes_pair
            for i in range(len(token_seq) - 1):
                if (token_seq[i], token_seq[i + 1]) == bytes_pair:
                    tok_seqs_with_bytes_pair.add(token_seq)
                    break
        bytes_pair_to_tokens[bytes_pair] = tok_seqs_with_bytes_pair
    return bytes_pair_to_tokens

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
    special_tokens: list[str],
    profile: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # ------------------------------------------------------------
    # step 1: init vocab
    # ------------------------------------------------------------
    with timer("init vocab", profile):
        vocab: dict[int, bytes] = {}
        vid = len(vocab)
        for token in special_tokens:
            vocab[vid] = token.encode("utf-8")
            vid += 1
        for i in range(256):
            vocab[vid] = bytes([i])
            vid += 1
        # print(f"vocab: {vocab}")
    
    # ------------------------------------------------------------
    # step 2: pre-tokenize
    # ------------------------------------------------------------
    with timer("pre-tokenization", profile):
        freq_table: dict[tuple[bytes], int] = {} # token_sequence → count
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
    
    # ------------------------------------------------------------
    # step 3: compute merges
    # ------------------------------------------------------------
    with timer("merge loop (total)", profile):
        merges: list[tuple(bytes, bytes)] = []
        # count byte-pair frequencies
        bytes_pair_counts = get_bytes_pair_counts(freq_table) # (bytes, bytes) → count
        bytes_pair_to_tokens = build_bytes_pair_to_tokens(freq_table, bytes_pair_counts, vocab_size) # (bytes, bytes) → set[token_sequence] 
        while vid < vocab_size: # merge until we reach vocab_size
            if not bytes_pair_counts: # no merge byte pairs to merge
                break
            # find the pair with max count
            pair = get_max_bytes_pair(bytes_pair_counts)

            # update freq_table (by merging the bytes pair), vocab, merges, 
            apply_bpe_merge(freq_table, pair, bytes_pair_counts, bytes_pair_to_tokens)
            
            merges.append(pair)
            vocab[vid] = pair[0]+pair[1]
            vid += 1
    return vocab, merges

if __name__ == "__main__":
    print(f"training on data from: {config["input_path"].split("/")[-1]}")
    vocab, merges = train_bpe(
        input_path=config["input_path"],
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        profile=True
    )
    # print(f"merges: {merges}")
    # print("\nvocab:")
    # for i in range(256, len(vocab)):
    #     print(f"{i}, {vocab[i]}")
    