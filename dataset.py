import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


PAD_ID = 0
def collate_fn(batch, tokenizer):
    '''
    batch = [
        (eng_tensor_1, hin_tensor_1),  
        (eng_tensor_2, hin_tensor_2),  
        ...
    ]

    These tensors need not be the same size. We want to make them the same size.
    '''
    eng_list = []
    hin_list = []

    for eng_tensor, hin_tensor in batch:
        eng_list.append(eng_tensor)
        hin_list.append(hin_tensor)
    
    eng_padded = pad_sequence(eng_list, batch_first=True, padding_value=PAD_ID)
    hin_padded = pad_sequence(hin_list, batch_first=True, padding_value=PAD_ID)

    return eng_padded, hin_padded



class TranslationDataset(Dataset):
    def __init__(self, data_path):
        # Load the pre-tokenized list of dictionaries
        self.data = torch.load(data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        eng_ids = item['eng_ids']
        hin_ids = item['hin_ids']

        # Adding <SOS>[1] and <EOS>[2]
        eng_tensor = torch.tensor([1] + eng_ids + [2])
        hin_tensor = torch.tensor([1] + hin_ids + [2])
    
        return eng_tensor, hin_tensor