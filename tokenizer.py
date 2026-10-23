import tiktoken

enc = tiktoken.get_encoding("gpt2")

vocab_size = enc.n_vocab

def encode(text):
    return enc.encode(text)

def decode(tokens):
    return enc.decode(tokens)