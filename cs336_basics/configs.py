config = {
    "tokenizer": {
        "save_path": "cs336_basics/data",
        "special_tokens": ["<|endoftext|>"],
        "PAT": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        
        # scenario: tokenizer experiments
        # "input_path": "cs336_basics/data/test_stories.txt",
        # "input_path": "cs336_basics/data/stress_test.txt",
        # "vocab_size": 256+100,
        
        # ( train_bpe_tinystories) scenario: run for tiny-stories dataset
        "input_path": "cs336_basics/data/TinyStoriesV2-GPT4-valid.txt",
        # "input_path": "cs336_basics/data/TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10000,
        
        # (train_bpe_expts_owt) scenario: run for open-web-text dataset
        # "input_path": "cs336_basics/data/owt_valid.txt",
        # "input_path": "cs336_basics/data/owt_train.txt",
        # "vocab_size": 32000,
        
    }
}