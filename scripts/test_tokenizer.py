from tokenizers import Tokenizer

# Load your trained tokenizer
tok = Tokenizer.from_file("tokenizer/bpe_5000/tokenizer.json")

# Sample text
text = "To be, or not to be: that is the question."

# Encode
encoded = tok.encode(text)
print("Input:", text)
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)

# Decode
decoded = tok.decode(encoded.ids)
print("Decoded back:", decoded)
