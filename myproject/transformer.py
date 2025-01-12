import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.onnx


import torchtext
from torchtext.data  import Field, TabularDataset, BucketIterator, Iterator
from torchtext import data

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np
import pandas as pd

import random
import math
import time
from tokenize import tokenize, untokenize
import io
from tqdm import tqdm

f = open("C:/Users/aaron/myproject/myproject/smartcontract.txt", "r")
file_lines = f.readlines()

dps = []
dp = None
for line in file_lines:
  if line[0] == "#":
    if dp:
      dp['solution'] = ''.join(dp['solution'])
      dps.append(dp)
    dp = {"question": None, "solution": []}
    dp['question'] = line[1:]
  else:
    dp["solution"].append(line)

#tokenize the solidity code and returns a tuple where each tuple contains a unique integer and string representation of a solidity token.
# UTF-8 because solidity extends past ASCII characters
# need to use io.BytesIO because tokenize import requires a file like object
# unique integer makes mapping easier

def tokenize_solidity_code(solidity_code):
    solidity_tokens = list(tokenize(io.BytesIO(solidity_code.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(solidity_tokens)):
        tokenized_output.append((solidity_tokens[i].type, solidity_tokens[i].string))
    return tokenized_output


# create a dataframe connecting the prompt with its solidity code

python_problems_df = pd.DataFrame(dps)
     
# Splitting data into 80% train and 20% validation

np.random.seed(0)
msk = np.random.rand(len(python_problems_df)) < 0.80 

train_df = python_problems_df[msk]
val_df = python_problems_df[~msk]



SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# tokenize the english prompts using spacy as it does not extend past ASCII characters

spacy_en = spacy.load("en_core_web_sm")

# Create two data fields for the enlgish prompt and solidity code

SRC = data.Field(tokenize=lambda text: [token.text for token in spacy_en.tokenizer(text)])

TRG = data.Field(tokenize = tokenize_solidity_code,
                    init_token='', 
                    eos_token='', 
                    lower=False)

fields = [('src', SRC),('trg', TRG)]

train_example = []
val_example = []



# preprocess code from the data frames in preparation for the transformer, fields sepcifies how to process data i.e. spacy or solidity tokenizer
# pass in case of an exception to skip potential data errors
#  

for i in range(train_df.shape[0]):
      try:
          ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
          train_example.append(ex)
      except:
          pass

for i in range(val_df.shape[0]):
    try:
        ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
        val_example.append(ex)
    except:
        pass       

#dataset is created for train and validation data.

train_data = data.Dataset(train_example, fields)
valid_data =  data.Dataset(val_example, fields)

# builds the vocab for the input and ouptut expected from the transformer
# minimum frequency isnt considered, so all tokens included
# all data in the training dataset is included in the vocab

SRC.build_vocab(train_data, min_freq = 0)
TRG.build_vocab(train_data, min_freq = 0)


print (TRG.vocab)

#Specify which device the transformer uses to run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

print(train_data[0].trg)

print(vars(train_data.examples[1]))

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

# Positionwise Feedforward Network
class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x)

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, device):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.layer_norm(src)

        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.layer_norm(src)
        return src

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, dropout, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, device) for _ in range(n_layers)])

    def forward(self, src, src_mask, d_model):
        src = self.tok_embedding(src) * math.sqrt(d_model)
        src = self.pos_embedding(src)
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.enc_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm(tgt)

        tgt2 = self.enc_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm(tgt)

        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.layer_norm(tgt)
        return tgt

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return self.output_layer(tgt)

# Complete Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, d_ff, n_encoder_layers, n_decoder_layers, dropout, device):
        super().__init__()
        self.encoder = TransformerEncoder(input_dim, d_model, n_heads, d_ff, n_encoder_layers, dropout, device)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ff, n_decoder_layers, dropout)
        self.src_word_emb = nn.Embedding(input_dim, d_model)
        self.tgt_word_emb = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.output_layer = nn.Linear(d_model, output_dim)
        self.device = device

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.positional_encoding(self.src_word_emb(src))
        tgt = self.positional_encoding(self.tgt_word_emb(tgt))
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask, src_mask)
        return self.output_layer(output)

# Training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(tqdm(iterator)):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg[:,:-1])
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Translation function
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    tokens = [token.lower() for token in spacy_en.tokenizer(sentence)]
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src)
            output = model.out(output)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)
        
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]

# Model, Optimizer, and Loss
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
N_LAYERS = 3
N_HEADS = 8
FF_DIM = 512
DROPOUT = 0.1

model = Transformer(INPUT_DIM, OUTPUT_DIM, HID_DIM, N_HEADS, FF_DIM, N_LAYERS, N_LAYERS, DROPOUT, device).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = TRG.vocab.stoi[TRG.pad_token])

# Train the model
N_EPOCHS = 10
CLIP = 1

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_data, optimizer, criterion, CLIP)
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')