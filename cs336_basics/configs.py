config = {
    "tokenizer": {
        # "input_path": "/Users/varmamzn/research/courses-conferences-programs/cs336-assignments/assignment1-basics/cs336_basics/data/test_stories.txt",
        # "input_path": "data/stress_test.txt",
        "input_path": "data/TinyStoriesV2-GPT4-valid.txt",
        "vocab_size": 256+1000,
        "special_tokens": ["<|endoftext|>"],
        "PAT": r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    }
}