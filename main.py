import torch
import pandas as pd

def main():
    print("Hello from rnn-translator!")
    print(torch.cuda.is_available())
        
    df = pd.read_csv("dataset/tatoeba.tsv", sep='\t', header=None)
    df.columns = ["eng_id", "english", "hin_id", "hindi"]
    df = df[["english", "hindi"]]
    MAX_LENGTH = 16
    df = df[df['english'].str.split().str.len() <= MAX_LENGTH]
    df = df[df['hindi'].str.split().str.len() <= MAX_LENGTH]

    print(f"Total valid sentence pairs: {len(df)}")
    print(df.head())

if __name__ == "__main__":
    main()
