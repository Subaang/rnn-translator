import torch
from tokenizers import Tokenizer
from nmt_model import NmtModel

def translate_author_style(model, sentence, eng_tokenizer, hin_tokenizer, device, max_length=50):
    model.eval()
    
    eng_encoded = eng_tokenizer.encode(sentence)
    eng_ids = [1] + eng_encoded.ids + [2]  # Add <SOS> and <EOS>
    eng_tensor = torch.tensor([eng_ids]).to(device) # (1, seq_len)
    
    hin_ids = [1] # <SOS> alone
    
    for _ in range(max_length):
        hin_tensor = torch.tensor([hin_ids]).to(device)
        
        with torch.no_grad():
            logits = model(eng_tensor, hin_tensor) # (batch_size, vocab_size, seq_len)
            last_step_logits = logits[0, :, -1] 
            predicted_id = last_step_logits.argmax(dim=-1).item() # Softmax already applied. This is just choosing most likely token
            
        hin_ids.append(predicted_id)
        
        if predicted_id == 2:
            break
            
    hindi_text = hin_tokenizer.decode(hin_ids[1:-1])
    return hindi_text


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    eng_tokenizer = Tokenizer.from_file("my_tokenizers/eng_tokenizer.json")
    hin_tokenizer = Tokenizer.from_file("my_tokenizers/hin_tokenizer.json")
    
    model = NmtModel(
        eng_vocab_size=eng_tokenizer.get_vocab_size(),
        hin_vocab_size=hin_tokenizer.get_vocab_size(),
        pad_id=0
    ).to(device)
    
    model.load_state_dict(torch.load("nmt_model_weights.pth", map_location=device, weights_only=True))
    
    print("\n=== Translator Ready ===")
    while True:
        english_input = input("English: ")
        if english_input.lower() == 'quit':
            break
        if not english_input.strip():
            continue
            
        translation = translate_author_style(model, english_input, eng_tokenizer, hin_tokenizer, device)
        print(f"Hindi:   {translation}\n")