import os
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
import numpy as np



class CodeBERT(nn.Module):   
    def __init__(self, encoder, config, tokenizer):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.dropout = nn.Dropout(0.5)
        
        
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
        
        
        
def model_setting_codebert(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    pretrain_model = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(pretrain_model)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)
    encoder = RobertaForSequenceClassification.from_pretrained(pretrain_model, config=config) 
    
    model = CodeBERT(encoder, config, tokenizer)

    model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.p))
    model.to(args.device) 
    model.eval()
            
    return model, tokenizer
        

class InputFeatures(object):
    def __init__(self, input_tokens, input_ids,):
        self.input_tokens = input_tokens
        self.input_ids = input_ids

class TextDataset():
    def __init__(self, code_list, tokenizer):
        self.dataset = []
        self.block_size = 512
        for code in code_list:
            code_tokens = tokenizer.tokenize(code)[:self.block_size-2]
            source_tokens = code_tokens
            source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
            padding_length = self.block_size-len(source_ids)
            source_ids += [tokenizer.pad_token_id]*padding_length
            input_feature = InputFeatures(source_tokens, source_ids)
            self.dataset.append(input_feature)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i): 
        return torch.tensor(self.dataset[i].input_ids)
    

def detect_codebert(snippet_dir, model, tokenizer, args):
    code_list = []
    for fn in os.listdir(snippet_dir):
        fp = os.path.join(snippet_dir, fn)
        with open(fp, 'r') as f:
            code = f.read()
        code_list.append(code)
    detect_dataset = TextDataset(code_list, tokenizer)
    detect_loader = DataLoader(detect_dataset, num_workers=4, batch_size=16, pin_memory=True)
    
    probs = []
    for data in detect_loader:
        inputs = data.to(args.device)
    
        with torch.no_grad():
            prob = model(input_ids=inputs, labels=None)
            probs.append(prob.cpu().numpy())
    
        probs = np.concatenate(probs, 0)
        preds = probs.argmax(-1)

        return preds
    