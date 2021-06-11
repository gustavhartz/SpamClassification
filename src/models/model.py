import torch
import torch.nn as nn

#from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTM(nn.Module):

    def __init__(self,tokenizer_vocab_size,hidden_dimension=128,embedding_dim = 50,text_len = 40,):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(tokenizer_vocab_size, embedding_dim,padding_idx=0)
        self.dimension = hidden_dimension
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(text_len*hidden_dimension, 1)

    def forward(self, text):

        text_emb = self.embedding(text)
        
        output,_ = self.lstm(text_emb)
        x = output.reshape(output.shape[0],-1)
        x = self.drop(x)
        
        return torch.sigmoid(self.fc(x))

        