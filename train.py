import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer 
from nmt_model import NmtModel
from dataset import TranslationDataset, collate_fn

def train(model, train_loader, optimizer, criterion, device, n_epochs=5):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0

        for eng_tensor, hin_tensor in train_loader:
            eng_tensor = eng_tensor.to(device)
            hin_tensor = hin_tensor.to(device)

            decoder_input = hin_tensor[:, :-1] 
            target_labels = hin_tensor[:, 1:]  

            optimizer.zero_grad()
            output = model(eng_tensor, decoder_input)
            loss = criterion(output, target_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{n_epochs}] | Average Loss: {avg_loss:.4f}")

def test(model, test_loader, criterion, device):
    model.eval()
    total_test_loss = 0
    
    with torch.no_grad():
        for eng_tensor, hin_tensor in test_loader:
            eng_tensor = eng_tensor.to(device)
            hin_tensor = hin_tensor.to(device)
            
            decoder_input = hin_tensor[:, :-1]
            target_labels = hin_tensor[:, 1:]
            
            output = model(eng_tensor, decoder_input)
            loss = criterion(output, target_labels)
            total_test_loss += loss.item()
            
    avg_test_loss = total_test_loss / len(test_loader)
    return avg_test_loss

if __name__ == "__main__":
    full_dataset = TranslationDataset("dataset/tokenized_tatoeba.pt")

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size 

    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=generator)

    print(f"Training pairs: {len(train_dataset)} | Test pairs: {len(test_dataset)}")
    
    BATCH_SIZE = 128 
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizers to determine vocabulary sizes...")
    eng_tokenizer = Tokenizer.from_file("my_tokenizers/eng_tokenizer.json")
    hin_tokenizer = Tokenizer.from_file("my_tokenizers/hin_tokenizer.json")
    
    ENG_VOCAB_SIZE = eng_tokenizer.get_vocab_size()
    HIN_VOCAB_SIZE = hin_tokenizer.get_vocab_size()
    print(f"English Vocab Size: {ENG_VOCAB_SIZE} | Hindi Vocab Size: {HIN_VOCAB_SIZE}")

    PAD_ID = 0

    model = NmtModel(
        eng_vocab_size=ENG_VOCAB_SIZE, 
        hin_vocab_size=HIN_VOCAB_SIZE, 
        pad_id=PAD_ID
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    print("\n--- Starting Training ---")
    train(model, train_loader, optimizer, criterion, device, n_epochs=15)

    print("\n--- Running Final Test ---")
    test_loss = test(model, test_loader, criterion, device) 
    print(f"Final Test Loss: {test_loss:.4f}")
    
    torch.save(model.state_dict(), "nmt_model_weights.pth")
    print("Model weights saved to nmt_model_weights.pth")