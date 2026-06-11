import torch
import pandas as pd
import tokenizers

def train_tokenizer(texts, save_path):
    bpe_model = tokenizers.models.BPE(unk_token='<unk>')
    bpe_tokenizer = tokenizers.Tokenizer(bpe_model)
    bpe_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    
    special_tokens = ['<pad>','<sos>','<eos>','<unk>']
    bpe_trainer = tokenizers.trainers.BpeTrainer(vocab_size=1000, special_tokens=special_tokens)
    
    print(f"Training tokenizer ({save_path})...")
    bpe_tokenizer.train_from_iterator(texts, trainer=bpe_trainer)
    
    bpe_tokenizer.save(save_path)
    print(f"Saved to {save_path}")

def tokenize():
    print("Loading data...")
    df = pd.read_csv("dataset/tatoeba.tsv", sep='\t', header=None)
    df.columns = ["eng_id", "english", "hin_id", "hindi"]
    df = df[["english", "hindi"]]
    
    df = df.dropna()

    eng_list = df["english"].astype(str).tolist()
    hin_list = df["hindi"].astype(str).tolist()

    train_tokenizer(eng_list, "my_tokenizers/eng_tokenizer.json")
    train_tokenizer(hin_list, "my_tokenizers/hin_tokenizer.json")

if __name__ == "__main__":
    tokenize()