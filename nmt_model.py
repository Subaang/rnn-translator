import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class NmtModel(nn.Module):
    def __init__(self, eng_vocab_size, hin_vocab_size, embed_dim=512, pad_id=0, hidden_dim=512, n_layers=2):
        super().__init__()
        self.pad_id = pad_id

        self.eng_embed = nn.Embedding(eng_vocab_size, embed_dim, padding_idx=pad_id)
        self.hin_embed = nn.Embedding(hin_vocab_size, embed_dim, padding_idx=pad_id)
        
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        
        # Output must project to the Target (Hindi) vocabulary size
        self.output = nn.Linear(hidden_dim, hin_vocab_size)

    def forward(self, eng_tensor, hin_tensor):
        source_embeddings = self.eng_embed(eng_tensor)
        target_embeddings = self.hin_embed(hin_tensor)

        source_lengths = (eng_tensor != self.pad_id).sum(dim=1)
        
        source_packed = pack_padded_sequence(
            source_embeddings, 
            lengths=source_lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        '''
        outputs: This contains the hidden state calculations for every single time step in the sequence, but only from the final layer of the GRU.
            In the Encoder: We throw this away using _, hidden_states because we don't care about the intermediate thoughts of the English sentence.
            In the Decoder: We keep this! These are the actual mathematical guesses for every word in the Hindi sentence. We pass this to the nn.Linear layer.


        hidden_states: This is the absolute final memory state at the very last time step, but it contains the memory from all layers of the GRU.
            In the Encoder: We keep this. This is your "Context Vector"—the mathematical summary of the entire English sentence.
            In the Decoder: We throw this away using outputs, _ because we are done generating the sentence and don't need to pass the memory to anything else.
        '''
        # If you do not pass a starting hidden state into a GRU, PyTorch automatically generates a matrix of pure zeros and feeds it in for you.
        _, hidden_states = self.encoder(source_packed)
        outputs, _ = self.decoder(target_embeddings, hidden_states)

        return self.output(outputs).permute(0, 2, 1)