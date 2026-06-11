import torch
import pandas as pd
from tokenizers import Tokenizer

def prepare_dataset():
    print("Loading data and tokenizers...")
    df = pd.read_csv("dataset/tatoeba.tsv", sep='\t', header=None)
    df.columns = ["eng_id", "english", "hin_id", "hindi"]
    df = df[["english", "hindi"]].dropna()

    eng_tokenizer = Tokenizer.from_file("my_tokenizers/eng_tokenizer.json")
    hin_tokenizer = Tokenizer.from_file("my_tokenizers/hin_tokenizer.json")


    #2 slots for <SOS> and <EOS>
    MAX_TOKENS = 48 
    
    processed_data = []
    dropped_count = 0

    print("Tokenizing and filtering...")
    

    for eng_text, hin_text in zip(df['english'], df['hindi']):
        
        eng_ids = eng_tokenizer.encode(str(eng_text)).ids
        hin_ids = hin_tokenizer.encode(str(hin_text)).ids
        
        if len(eng_ids) <= MAX_TOKENS and len(hin_ids) <= MAX_TOKENS:
            processed_data.append({
                'eng_ids': eng_ids,
                'hin_ids': hin_ids
            })
        else:
            dropped_count += 1

    save_path = "dataset/tokenized_tatoeba.pt"
    torch.save(processed_data, save_path)
    
    print("\n--- Pre-processing Complete ---")
    print(f"Valid pairs saved: {len(processed_data)}")
    print(f"Pairs dropped:     {dropped_count}")
    print(f"Saved to:          {save_path}")

if __name__ == "__main__":
    prepare_dataset()