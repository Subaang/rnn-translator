import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NmtModel(nn.Module):
    def __init__(self, eng_vocab_size, hin_vocab_size, embed_dim=512, pad_id=0, hidden_dim=512, n_layers=2):
        super().__init__()
        self.pad_id = pad_id

        self.eng_embed = nn.Embedding(eng_vocab_size, embed_dim, padding_idx=pad_id)
        self.hin_embed = nn.Embedding(hin_vocab_size, embed_dim, padding_idx=pad_id)
        
        self.encoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.decoder = nn.GRU(embed_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        
        self.attention_linear = nn.Linear(hidden_dim * 2, hin_vocab_size) # [batch,seq_len_dec,1024] -> [batch,seq_len_dec,hin_vocab_size]


    def attention(self, query, key, value):
        d_k = query.size(-1) # == 512
        scores = (query @ key.transpose(1, 2)) / (d_k ** 0.5) # [batch, seq_len_dec, 512] @ [batch, 512, seq_len_enc] -> [batch, seq_len_dec, seq_len_enc]
        weights = torch.softmax(scores, dim=-1)# -> [batch, seq_len_dec, seq_len_enc]. for each token in the decoder, we get the prob distribution of encoder input tokens
        return weights @ value # [batch, seq_len_dec, seq_len_enc] @ [batch, seq_len_enc, 512] -> [batch, seq_len_dec, 512]
                                

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
        
        encoder_outputs_packed, encoder_hidden_states = self.encoder(source_packed)
        decoder_outputs, _ = self.decoder(target_embeddings, encoder_hidden_states)
        encoder_outputs, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)
        attention_outputs = self.attention(query=decoder_outputs, key=encoder_outputs, value=encoder_outputs)
        combined_output = torch.cat((attention_outputs, decoder_outputs), dim=-1)

        # Predict using the linear layer and permute for CrossEntropyLoss [batch, classes, seq_len]
        return self.attention_linear(combined_output).permute(0, 2, 1)