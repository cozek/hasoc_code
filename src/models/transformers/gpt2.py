import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import GPT2Tokenizer, GPT2Model

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str, 
                 cache_dir:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(
            gpt_model_name, cache_dir = cache_dir
        )
        self.fc1 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x_in):
        
        gpt_out = self.gpt2model(x_in)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        prediction_vector = self.fc1(gpt_out.view(batch_size,-1)) #(batch_size , max_len, num_classes)
    
        return prediction_vector
        