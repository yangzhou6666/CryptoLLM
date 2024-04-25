import torch
import torch.nn as nn

    
    
class ELECTRA(nn.Module):   
    def __init__(self, encoder, config, tokenizer, args):
        super(ELECTRA, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args
        self.dropout = nn.Dropout(args.dropout_prob)
        
        
    def forward(self, input_ids=None, labels=None):
        logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        logits = self.dropout(logits)
        prob = torch.sigmoid(logits)
                
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(prob[:,1], labels.float())
            return loss, prob
        else:
            return prob