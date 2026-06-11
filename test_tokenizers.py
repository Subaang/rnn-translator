import tokenizers

def test_tokenizers():
    eng_tokenizer = tokenizers.Tokenizer.from_file('my_tokenizers/eng_tokenizer.json')
    encoded = eng_tokenizer.encode("Hello world")
    print(encoded.tokens)

if __name__ == "__main__":
    test_tokenizers()