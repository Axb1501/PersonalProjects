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

     
# Getting the data from smartcontract.txt file

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

Input = data.Field(tokenize=lambda text: [token.text for token in spacy_en.tokenizer(text)])

Output = data.Field(tokenize = tokenize_solidity_code,
                    init_token='', 
                    eos_token='', 
                    lower=False)

fields = [('Input', Input),('Output', Output)]

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

Input.build_vocab(train_data, min_freq = 0)
Output.build_vocab(train_data, min_freq = 0)


print (Output.vocab)

#Specify which device the transformer uses to run

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

print(train_data[0].Output)

print(vars(train_data.examples[1]))


#Encoder processes input sequences for th transformer
# Initialize token embeddings, positional encoding, layers, dropout and scale

#Encoder forward pass takes input sequence, and tensor used to remove padding tokens from self attention
# pos generates positional indices
# apply token embedding to src and positional embeddings 
# scale embeddings to prevent self attention mechanisms being too small or too big
# dropout to prevent overfitting
# go through your encoder layers then return the src

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 1000):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

# represents a single layer of the encoder
# intiliaze encoder layer and several features 
# Self attention mechanism computes self attentyion scores and applies it to the input. weights the importance of words
# residual connection by adding self attention to input sequence, and the output of the ff network and self attention layer. helps the gradient flow
# Feedforward network. capture non linear relationships
# layer normalization to sum of ouptut of self attention layer and dropuout, reduc variance by normalizing
# return processed src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

# Initialize the model
# Change the input tensor x dimensionality of hid_dim to pf_dim
# ReLU for non-linearity
# then transform tensor back to original dimensionality

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x




class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, num_heads, dropout, device):
        super(MultiHeadAttentionLayer, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert hid_dim % num_heads == 0, "hid_dim must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = hid_dim # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = hid_dim // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(hid_dim, hid_dim) # Query transformation
        self.W_k = nn.Linear(hid_dim, hid_dim) # Key transformation
        self.W_v = nn.Linear(hid_dim, hid_dim) # Value transformation
        self.W_o = nn.Linear(hid_dim, hid_dim) # Output transformation
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        attn_probs = self.dropout(attn_probs)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
        
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, hid_dim = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention and obtain attention scores
        attn_output, attn_scores = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output, attn_scores

# Initialize module
# token embeddings, positional embeddings, linear layer, dropout, scaling
# linear layer for ouptut transformation
# trg mask masks future tokens and src mask removes padding
#enc_src encoded source sequence
# transform output to match dimesnionality of output vocab

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim,    
                 hid_dim,      
                 n_layers,      
                 n_heads,       
                 pf_dim,        
                 dropout,      
                 device,       
                 max_length=10000):  #
        super().__init__()

        self.device = device

        # Token embedding layer
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        # Positional embedding layer
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        # List of decoder layers
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        
        # Output linear layer
        self.fc_out = nn.Linear(hid_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
    
        batch_size = trg.shape[0]   # Batch size
        trg_len = trg.shape[1]       # Length of the target sequence
        
        # Generate positional indices for the target sequence
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # Add positional embeddings to token embeddings and apply dropout
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg = [batch size, trg len, hid dim]

        # Pass the target sequence through each decoder layer
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        # Apply output linear layer
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
            
        return output, attention



class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        
        # Layer normalization for self-attention mechanism
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        # Layer normalization for encoder-decoder attention mechanism
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        # Layer normalization for feedforward network
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        
        # Multi-head self-attention mechanism
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        # Multi-head attention mechanism for encoder-decoder attention
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        # Position-wise feedforward network
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
    
        # Self-attention mechanism
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        # Dropout, residual connection, and layer normalization for self-attention
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        # Encoder-decoder attention mechanism
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        # Dropout, residual connection, and layer normalization for encoder-decoder attention
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        # Position-wise feedforward network
        _trg = self.positionwise_feedforward(trg)
        
        # Dropout, residual connection, and layer normalization for feedforward network
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        # Return output tensor and attention scores
        return trg, attention



class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # Generate mask tensor to mask out padding tokens in source sequences.
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    

    # Generate mask tensor to mask out padding tokens and future tokens in target sequences.

    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask


    #passes through the encoder to obtain encoded source representations
    # passes the target sequence  and masks to the decoder to generate output predictions and attention scores for model output

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention 

INPUT_DIM = len(Input.vocab)
OUTPUT_DIM = len(Output.vocab)
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)

dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

SRC_PAD_IDX = Input.vocab.stoi[Input.pad_token]
TRG_PAD_IDX = Output.vocab.stoi[Output.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# initialize weights using a uniform distribution for the model and apply them
# optimize model and establish a learning rate for optimizer

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights);

LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# initiliaze parameters
# smoothing to prevent model overconfidence
# nll loss and cross entropy loss perform log likelihood for different inputs

class SparseCategoricalLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(SparseCategoricalLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, inputs, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist

        smooth_eps = self.smooth_eps or 0

        # Ordinary log-likelihood - use cross_entropy from nn
        if self.from_logits:
            loss = F.cross_entropy(inputs, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        else:
            loss = F.nll_loss(inputs, target, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction)

        # does smoothing if coeficcient exceeds 1 and hasnt been smoothed
        if smooth_eps > 0 and smooth_dist is not None:
            num_classes = inputs.size(-1)

            # convert class indices to one hot encoded versions, ensures each class represented by unique binary vector. input feature and class label relation
            if isinstance(target, torch.LongTensor):
                target = F.one_hot(target, num_classes=num_classes).float()

            # smooths on hot encoded of target tensor
            target = target.lerp(smooth_dist, smooth_eps)

            #Kullback loss
            loss = (1.0 - smooth_eps) * loss + smooth_eps * F.kl_div(F.log_softmax(inputs, dim=-1), target, reduction='batchmean')

        return loss


# calculate the loss based on given input predictions and target labels.
# calls the loss gunction and counts non pad element
# return loss and count

def maskNLLLoss(inp, target, mask):
    # print(inp.shape, target.shape, mask.sum())
    nTotal = mask.sum()

    # Instantiate SparseCategoricalLoss with your desired parameters
    sparseCrossEntropy = SparseCategoricalLoss(ignore_index=TRG_PAD_IDX, smooth_eps=0.20)

    # Calculate loss using SparseCategoricalLoss
    loss = sparseCrossEntropy(inp, target)
    loss = loss.to(device)

    return loss, nTotal.item()

criterion = maskNLLLoss

# generates a mask to hide future tokens in target sequences
# create a mask tensor and mask for future tokens then combines them
# returns mask

def make_trg_mask(trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != TRG_PAD_IDX).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

# trains the model
# calls previous functions and runs them
# iterates over every batch backproopogation to compute gradients
# clip gradients to prevent high gradients
# optimizer updates model
# criterion computes loss
# transpose src and trg to match model format

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    n_totals = 0
    print_losses = []
    for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
        # print(batch)
        loss = 0
        src = batch.Input.permute(1, 0)
        trg = batch.Output.permute(1, 0)
        trg_mask = make_trg_mask(trg)
        optimizer.zero_grad()
        
        #predictions based on src and trg sequences
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
        # obtain model predicitons
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        mask_loss, nTotal = criterion(output, trg, trg_mask)
        
        mask_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal


        
    return sum(print_losses) / n_totals

# evaluates the loss on the dataset
# set model to eval mode
# Iterate over batches
# do the same as train
# return the average loss

def evaluate(model, iterator, criterion):
    model.eval()
    
    n_totals = 0
    print_losses = []
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator), total=len(iterator)):
            src = batch.Input.permute(1, 0)
            trg = batch.Output.permute(1, 0)
            trg_mask = make_trg_mask(trg)

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            mask_loss, nTotal = criterion(output, trg, trg_mask)

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Handle the case where n_totals is zero to avoid division by zero
    if n_totals == 0:
        return 0

    return sum(print_losses) / n_totals

# use epochs for training
# epoch time taken for each epoch
# choose the best model by updating the best valid loss
# bucket iterator retrieves training and validation iterators
# train function and evaluation fucntion is then used to compute the training and eval loss

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 50
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    BATCH_SIZE = 16
    train_iterator, valid_iterator = BucketIterator.splits((train_data, valid_data), batch_size = BATCH_SIZE, 
                                                                sort_key = lambda x: len(x.Input),
                                                                sort_within_batch=True, device = device)

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} ')
    print(f'\t Val. Loss: {valid_loss:.3f} ')

SRC = Input
TRG = Output


# tokenize input and convert it all to lowercase format, if not a string assume tokenization
# add token padding start and end sequence
# convert tokens to indexes and then to tensors for the transformer
# source mask generated
# encode source sequence
# predict tokens next in sequence and terminate once at end token
# reconvert the indexes to tokens and return the transalted tokens

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50000):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention

# split into list of words
# translate english to solidity
# untokenize tokens from translate function
# return code

def eng_to_solidity(src):
  src=src.split(" ")
  translation, attention = translate_sentence(src, SRC, TRG, model, device)

  return untokenize(translation[:-1]).decode('utf-8')






