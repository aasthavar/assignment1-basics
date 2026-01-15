# optimization of code inspiration from here: https://github.com/thtfive/cs336-assignment1-basics/blob/thtfive/basics/cs336_basics/train_bpe.py#L177

import regex as re
from .configs import config
import time
from contextlib import contextmanager
import multiprocessing as mp
from .pretokenization_example import find_chunk_boundaries
from .common import (
    write_vocab_to_file, 
    write_merges_to_file,
    read_merges_from_file,
    read_vocab_from_file
)
from tqdm import tqdm
import json
import psutil, os

from tests.common import FIXTURES_PATH

config = config["tokenizer"]
config["PAT"] = re.compile(config["PAT"])

@contextmanager
def timer(name: str, enabled: bool):
    if not enabled:
        yield
        return
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[TIMER] {name}: {end - start:.4f}s")

@contextmanager
def rss_timer(name: str, enabled: bool):
    if not enabled:
        yield
        return
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss
    yield
    after = process.memory_info().rss
    print(f"[RSS] {name}: Δ={(after-before)/1e6:.2f}MB")


def apply_bpe_merge(
    freq_table: dict[tuple[bytes, ...], int],
    max_pair: tuple[bytes, bytes],
    bytes_pair_counts: dict[tuple[bytes, bytes], int],
    bytes_pair_to_tokens: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> None:
    """
    Apply one BPE merge step for `max_pair`.

    This function represents the *final, fully optimized* version.
    All updates are done in-place.

    ──────────────────────────────────────────────────────────────
    Optimization history (conceptual):

    Version 0 (naive, O(N)):
        - Loop over *all* token sequences
        - Merge `max_pair` wherever it appears
        - Rebuild a new freq_table from scratch

    Version 1 (localized update):
        - Maintain bytes_pair_to_tokens
        - Only touch token sequences known to contain `max_pair`
        - Still returned a new freq_table

    Version 2 (incremental, current):
        - Mutate freq_table in-place
        - Incrementally update:
            * freq_table
            * bytes_pair_counts
            * bytes_pair_to_tokens
        - No global recomputation of bytes_pair_counts, bytes_pair_to_tokens
        - No return value

    The current implementation corresponds to Version 2.
    ──────────────────────────────────────────────────────────────
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


def pretokenize_chunk(params) -> dict[tuple[bytes, ...], int]:
    """
    function does the following:
    - reads a chunk of the file
    - removes special tokens
    - applies pre-tokenization regex
    - converts text -> byte-level token seqences
    - builds and returns a local frequency table
    """
    start, end = params["start"], params["end"]
    freq_table: dict[tuple[bytes], int] = {} # token_sequence → count
    with open(params["file_name"], "rb") as f:
        f.seek(start)
        text = f.read(end-start).decode("utf-8", errors="ignore")
        escaped = [re.escape(token) for token in params["special_tokens"]]
        documents = re.split("|".join(escaped), text)
        for doc in documents:
            words = config["PAT"].findall(doc)
            for word in words:
                key = tuple([bytes([x]) for x in word.encode("utf-8")])
                freq_table[key] = freq_table.get(key, 0) + 1
    return freq_table


def merge_freq_tables(
    freq_tables:list[dict[tuple[bytes, ...], int]]
) -> dict[tuple[bytes, ...], int]:
    """
    merge frequencies from multiple chunks
    """
    merged_freq_table = {}
    for table in freq_tables:
        for key, count in table.items():
            merged_freq_table[key] = merged_freq_table.get(key, 0) + count
    return merged_freq_table


def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str],
    split_special_token: str = "<|endoftext|>",
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
    with timer("pre-tokenization", profile), rss_timer("pre-tokenization", profile):
        num_processes = mp.cpu_count()
        # print(f"num_processes for parallelizing pretokenization: {num_processes}")
        with open(input_path, "rb") as f:
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=num_processes,
                split_special_token=split_special_token.encode("utf-8")
            )
        with mp.Pool(processes=num_processes) as pool:
            tasks = [
                {
                    "file_name": input_path, 
                    "start": chunk_boundaries[i], 
                    "end": chunk_boundaries[i+1], 
                    "special_tokens": special_tokens
                } for i in range(len(chunk_boundaries)-1)
            ]
            freq_tables = pool.map(pretokenize_chunk, tasks)
            freq_table = merge_freq_tables(freq_tables)
    
    # ------------------------------------------------------------
    # step 3: compute merges
    # ------------------------------------------------------------
    with timer("compute merges (total)", profile), rss_timer("compute merges (total)", profile):
        
        with timer("compute merges: calc counts", profile), rss_timer("compute merges: calc counts", profile):
            # count byte-pair frequencies
            bytes_pair_counts = get_bytes_pair_counts(freq_table) # (bytes, bytes) → count
            # find all sequences which contain byte_pair and push it to dict
            bytes_pair_to_tokens = build_bytes_pair_to_tokens(freq_table, bytes_pair_counts, vocab_size) # (bytes, bytes) → set[token_sequence] 
        
        iterator = range(vid, vocab_size)
        if profile: # using this time slightly increased
            iterator = tqdm(
                iterator,
                desc="[TIMER] compute merges: bpe merge",
                total=vocab_size - vid,
                unit="merge",
            )
        merges: list[tuple(bytes, bytes)] = []
        
        for _ in iterator: # while vid < vocab_size: # merge until we reach vocab_size
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

def test_read_write_vocab_merges():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    
    # Write vocab and merges to files
    write_vocab_to_file(vocab, "vocab.json")
    write_merges_to_file(merges, "merges.txt")
    
    # Read back from files
    vocab2 = read_vocab_from_file("vocab.json")
    merges2 = read_merges_from_file("merges.txt")
    
    assert vocab == vocab2
    assert merges == merges2
        
if __name__ == "__main__":
    
    # test_read_write_vocab_merges()
    
    # NOTE: For train_bpe_tinystories, train_bpe_expts_owt: just change the configs.py
    filename_with_ext = config["input_path"].split("/")[-1]
    print(f"config: {json.dumps(config, indent=2, default=str)}")
    print(f"training on data: {filename_with_ext}")
    
    print("\n-------- start --------")
    vocab, merges = train_bpe(
        input_path=config["input_path"],
        vocab_size=config["vocab_size"],
        special_tokens=config["special_tokens"],
        split_special_token="<|endoftext|>",
        profile=True
    )
    print("\n-------- artifacts --------")
    filename = filename_with_ext.split(".")[0]
    
    vocab_file_path = f"{config['save_path']}/{filename}_vocab.json"
    print(f"vocab file saved as: {vocab_file_path}")
    write_vocab_to_file(vocab, vocab_file_path)
    
    merges_file_path = f"{config['save_path']}/{filename}_merges.txt"
    print(f"merges file saved as: {merges_file_path}")
    write_merges_to_file(merges, merges_file_path)

    
    # NOTE; missed noting the baseline numbers - it was >> 15s for merge loop
    
    # NOTE: after optimizing merge loop step
    # training on data: TinyStoriesV2-GPT4-valid.txt
    # [TIMER] init vocab: 0.0000s
    # [TIMER] pre-tokenization: 4.7191s
    # [TIMER] merge loop (total): 5.2906s
    
    # NOTE: after optimizing pretokenization step
    # training on data from: TinyStoriesV2-GPT4-valid.txt
    # [TIMER] init vocab: 0.0000s
    # num_processes for parallelizing pretokenization: 10
    # [TIMER] pre-tokenization: 0.8908s
    # [TIMER] merge loop (total): 5.4502s
    
    
    # training on data: TinyStoriesV2-GPT4-train.txt
    # [TIMER] init vocab: 0.0000s
    # [TIMER] pre-tokenization: 78.9581s
    # [TIMER] compute merges: calc counts: 50.0580s
    # [TIMER] compute merges: bpe merge: 100%|█████████████████████████████████████████████████████████████████████████████| 9743/9743 [00:31<00:00, 312.30merge/s]
    # [TIMER] compute merges (total): 81.2624s
    
    
    # training on data: owt_valid.txt
    # [TIMER] init vocab: 0.0000s
    # [TIMER] pre-tokenization: 7.9842s
    # [TIMER] compute merges: calc counts: 2798.4712s
    # [TIMER] compute merges: bpe merge: 100%|████████████████████████████████████████████████████████████████████████████| 31743/31743 [17:16<00:00, 30.62merge/s]
    # [TIMER] compute merges (total): 3835.1516s