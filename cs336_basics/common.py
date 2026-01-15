from tests.common import gpt2_bytes_to_unicode
import json

def write_vocab_to_file(vocab: dict[int, bytes], file_path: str):
    gpt2_byte_decoder = gpt2_bytes_to_unicode()

    token_to_id = {}
    for idx, byte_seq in vocab.items():
        token_str = "".join(gpt2_byte_decoder[b] for b in byte_seq)
        token_to_id[token_str] = idx

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=2)


def read_vocab_from_file(file_path: str) -> dict[int, bytes]:
    gpt2_byte_encoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(file_path, encoding="utf-8") as f:
        token_to_id = json.load(f)

    vocab: dict[int, bytes] = {}
    for token_str, idx in token_to_id.items():
        token_bytes = bytes(gpt2_byte_encoder[ch] for ch in token_str)
        vocab[idx] = token_bytes

    return vocab


def write_merges_to_file(merges, file_path):
    gpt2_byte_decoder = {k: v for k, v in gpt2_bytes_to_unicode().items()}
    human_readable_merges = [
        (
            "".join([gpt2_byte_decoder[token] for token in merge_token_1]),
            "".join([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in merges
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for (a, b) in human_readable_merges:
            f.write(a + " " + b)
            f.write("\n")


def read_merges_from_file(file_path):
    gpt2_byte_encoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(file_path, encoding="utf-8") as f:
        data = f.readlines()
    merges = []
    for line in data:
        token1, token2 = line.strip().split(" ", 1)
        tokens_in_bytes = bytes([gpt2_byte_encoder[ch] for ch in token1]), bytes([gpt2_byte_encoder[ch] for ch in token2])
        merges.append((tokens_in_bytes[0], tokens_in_bytes[1]))
    return merges