import regex as re
from typing import Iterable, Iterator

from .common import read_vocab_from_file, read_merges_from_file
from .configs import config

config = config["tokenizer"]
config["PAT"] = re.compile(config["PAT"])


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.bytes_to_id = {v: k for k, v in vocab.items()}
        
        self.merges = merges
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens or []
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                idx = len(self.vocab)
                self.vocab[idx] = token_bytes
                self.bytes_to_id[token_bytes] = idx
        special_tokens_sorted = sorted(
            self.special_tokens, key=len, reverse=True
        )
        self.special_tokens_pattern = "(" + "|".join(map(re.escape, special_tokens_sorted)) + ")"


    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        vocab = read_vocab_from_file(vocab_filepath)
        merges = read_merges_from_file(merges_filepath)
        return cls(vocab, merges, special_tokens)
    
    
    def _apply_bpe_merges(self, pre_token: list[bytes]) -> list[int]:
        while True:
            candidates = []
            # print(f"pre_token: {pre_token}")
            # print(f"merges_rank: {self.merges_rank}")
            for i in range(len(pre_token)-1):
                pair = (pre_token[i], pre_token[i+1])
                # print(f"pair: {pair}")
                if pair in self.merges_rank:
                    candidates.append(pair)
            # print(f"candidates: {candidates}")
            if not candidates:
                break
            
            # choose lowest rank (earliest trained) merge 
            best_pair = min(candidates, key=lambda p: self.merges_rank[p])
            
            a, b = best_pair
            merged = a + b
            # print(f"best_pair: {best_pair}, merged: {merged}")
            
            
            new_pre_token = []
            i = 0
            L = len(pre_token)
            
            while i < L:
                # print(f"{i}: {pre_token[i]}")
                if i+1<L and (pre_token[i], pre_token[i+1]) == best_pair:
                    new_pre_token.append(merged)
                    # print(f"{i}: yes: {merged}")
                    i += 2
                else:
                    new_pre_token.append(pre_token[i])
                    # print(f"{i}: no: {pre_token[i]}")
                    i += 1
            # print(f"new_pre_token: {new_pre_token}")
            pre_token = new_pre_token
            # print(f"-"*10)
        
        return [self.bytes_to_id[b] for b in pre_token]
    
    
    def encode(
        self, 
        text: str
    ) -> list[int]:
        token_ids: list[int] = []
        
        if not self.special_tokens:
            parts = [text]
        else:
            
            parts = re.split(self.special_tokens_pattern, text)
        print(f"parts: {parts}")
        
        for part in parts:
            if not part:
                continue
            
            # special token
            if part in self.special_tokens:
                token_ids.append(self.bytes_to_id[part.encode("utf-8")])
                continue

            # normal text
            words = config["PAT"].findall(part)
            for word in words:
                # print(f"word: {word}")
                pre_token = [bytes([b]) for b in word.encode("utf-8")]
                token_ids.extend(self._apply_bpe_merges(pre_token))

        return token_ids
    
    
    def encode_iterable(
        self, 
        iterable: Iterable[str]
    ) -> Iterator[int]:
        remaining_text = ""
        for text in iterable:
            text = remaining_text + text
            
            matches = list(config["PAT"].finditer(text))
            match = matches[-1] if matches else None
            
            if match:
                text = text[:match.end()]
                remaining_text = text[match.end():]
                for id in self.encode(text):
                    yield id
            else:
                remaining_text = text
       
        if remaining_text:
            for id in self.encode(remaining_text):
                yield id
    
    def decode(
        self, 
        ids: list[int]
    ) -> str:
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")
    
    
def test_1():
    vocab = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    special_tokens = None
    
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    
    text = "the cat ate"
    # text = " ate"
    ids = tokenizer.encode(text)
    print("ids:", ids)
    print("decoded:", tokenizer.decode(ids))
    assert ids == [9, 7, 1, 5, 10, 3]


def test_2():
    vocab = {0: b"<|endoftext|>", 1: b"<|startoftext|>", 2: b"ab", 3: b"a", 4: b"b", 5: b"c"}
    merges = [(b"a", b"b"), (b"ab", b"c")]
    special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    
    text = "<|startoftext|>abacaba<|endoftext|>"
    ids = tokenizer.encode(text)    
    print("ids:", ids)
    print("decoded:", tokenizer.decode(ids))
    assert ids == [1, 2, 3, 5, 2, 3, 0]


def test_3():
    vocab_path = "cs336_basics/data/TinyStoriesV2-GPT4-train_vocab.json"
    merges_path = "cs336_basics/data/TinyStoriesV2-GPT4-train_merges.txt"
    special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens
    )
    text = "<|startoftext|>abacaba<|endoftext|>"
    ids = tokenizer.encode(text)    
    print("ids:", ids)
    print("decoded:", tokenizer.decode(ids))
    
    
if __name__ == "__main__":
    test_1()
    
    print(f"-"*10)
    test_2()
    
    print(f"-"*10)
    test_3()