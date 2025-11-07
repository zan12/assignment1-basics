import itertools
from collections import Counter
from functools import partial
import os
import regex as re
import multiprocessing as mp
from .timer import Timer
import pickle


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pre_tokenize(input_path: str, special_tokens: list[str]) -> list[str]:
    with open(input_path, "r") as f:
        text = f.read()
    # Special tokens partition documents.
    docs = re.split("|".join(special_tokens), text)
    words = []
    for doc in docs:
        words += re.findall(PAT, doc)
    return words


def pre_tokenize_chunk(
    chunk: tuple[int, int],
    input_path: str,
    special_tokens: list[str],
):
    start, end = chunk
    
    words = []
    with open(input_path, "rb") as f:
        f.seek(start)
        # Decode will not break a character, since we stop at the special token boundary.
        text_chunk = f.read(end-start).decode("utf-8")
        # Special tokens partition documents.
        docs = re.split("|".join(special_tokens), text_chunk)
        for doc in docs:
            words.extend(re.findall(PAT, doc))
    return words
    

def better_pre_tokenize(
    input_path: str, 
    special_tokens: list[str],
    num_of_processes: int = 1,
) -> list[str]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        chunk_size = file_size // num_of_processes # roughly estimate the chunk size
        slice_size = 4_096
        boundaries = [i*chunk_size for i in range(num_of_processes+1)]
        boundaries[-1] = file_size

        special_token_bytes = special_tokens[0].encode("utf-8")
        
        for i in range(1, num_of_processes):
            f.seek(boundaries[i])
            while True:
                slice = f.read(slice_size)
                if slice == b'': # EOF
                    boundaries[i] = file_size
                    break
                find_at = slice.find(special_token_bytes)
                if find_at != -1:
                    boundaries[i] += find_at
                    break
                boundaries[i] += slice_size
        
        words = []
        # # serialize pre-tokenize
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     f.seek(start)
        #     text_chunk = f.read(end-start).decode("utf-8")
        #     # Special tokens partition documents.
        #     docs = re.split("|".join(special_tokens), text_chunk)
        #     for doc in docs:
        #         words += re.findall(PAT, doc)

        # parallelize pre-tokenize
        chunks = zip(boundaries[:-1], boundaries[1:])
        worker = partial(
            pre_tokenize_chunk, 
            input_path=input_path,
            special_tokens=special_tokens,
        )
        with mp.Pool(processes = num_of_processes) as pool:
            results = pool.map(worker, chunks)
        words = list(itertools.chain.from_iterable(results))
        
    return words
    

def word_to_bytes(word: str) -> list[bytes]:
    indices = list(word.encode("utf-8"))
    return tuple([bytes([b]) for b in indices])


def count_words_bytes_freq(word_freq: dict[str, int]):
    words_bytes_freq = Counter()
    for word, freq in word_freq.items():
        indices = list(word.encode("utf-8"))
        bytes_ = tuple([bytes([b]) for b in indices])
        words_bytes_freq[bytes_] = freq
    return words_bytes_freq


def count_pair_freq(bytes_freq: dict[tuple, int]):
    byte_pair_freq = Counter()
    for bytes_, freq in bytes_freq.items():
        for i in range(len(bytes_)-1):
            byte_pair_freq[(bytes_[i], bytes_[i+1])] += freq
    return byte_pair_freq


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # words = pre_tokenize(input_path, special_tokens) # [w00,w01,..., w10,w11,...]
    words = better_pre_tokenize(input_path, special_tokens, 10) # [w00,w01,..., w10,w11,...]
    word_freq = Counter(words) # {w0: 1, w1: 2, ...}, wi is a string / word
    words_bytes_freq = count_words_bytes_freq(word_freq) # {b0: 1, b1: 2, ...}, bi is a tuple of bytes
    pair_freq = count_pair_freq(words_bytes_freq)
    most_freq_pair = max(list(pair_freq.items()), key=lambda x: (x[1],x[0]))[0]

    # Initialize the vocabulary with single bytes and special tokens.
    init_vocab_size = 256
    vocab = dict([(i, bytes([b])) for i, b in enumerate(range(init_vocab_size))])
    for i, st in enumerate(special_tokens):
        vocab[init_vocab_size+i] = st.encode("utf-8")

    merges = []
    for iter in range(init_vocab_size+len(special_tokens), vocab_size):
        print("Iter: ", iter)
        # Update merges and vocab
        merges.append(most_freq_pair)
        combined_bytes = most_freq_pair[0] + most_freq_pair[1]
        vocab[iter] = combined_bytes
        new_words_bytes_freq = Counter()
        for word_bytes, freq in words_bytes_freq.items():
            if len(word_bytes) > 1:
                bytes_pairs = list(zip(list(word_bytes)[:-1], list(word_bytes)[1:]))
                i, end = 0, len(bytes_pairs)
                if most_freq_pair not in bytes_pairs:
                    new_words_bytes_freq[word_bytes] = words_bytes_freq[word_bytes]
                    continue
                while i < end:
                    if bytes_pairs[i] == most_freq_pair:
                        pair_freq[most_freq_pair] -= freq
                        if i > 0:
                            pair_freq[bytes_pairs[i-1]] -= freq
                            bytes_pairs[i-1] = (bytes_pairs[i-1][0], combined_bytes)
                            pair_freq[bytes_pairs[i-1]] += freq
                        if i < end-1:
                            pair_freq[bytes_pairs[i+1]] -= freq
                            bytes_pairs[i+1] = (combined_bytes, bytes_pairs[i+1][1])
                            pair_freq[bytes_pairs[i+1]] += freq
                        bytes_pairs.pop(i)
                        end -= 1
                    i += 1
                if len(bytes_pairs) < 1:
                    word_bytes = (combined_bytes,)
                else:
                    word_bytes = tuple([bytes_pairs[0][0]] + [i[1] for i in bytes_pairs])
            new_words_bytes_freq[word_bytes] += freq
        words_bytes_freq = new_words_bytes_freq
        most_freq_pair = max(list(pair_freq.items()), key=lambda x: (x[1],x[0]))[0]
    return vocab, merges


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab, merges = train_bpe(
        input_path=os.path.join(current_dir, "../data/TinyStoriesV2-GPT4-valid.txt"),
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    vocab_file_path = os.path.join(current_dir, "../ckpts/TinyStoriesV2-GPT4-tokenizer-valid-vocab.pkl")
    merges_file_path = os.path.join(current_dir, "../ckpts/TinyStoriesV2-GPT4-tokenizer-valid-merges.pkl")
    with open(vocab_file_path, "wb") as fv:
        pickle.dump(vocab, fv)
    with open(merges_file_path, "wb") as fm:
        pickle.dump(merges, fm)