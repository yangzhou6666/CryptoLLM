import os
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import DataLoader
import numpy as np



class CodeT5(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(CodeT5, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(0.5)

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids[:512]
        vec = self.get_t5_vec(source_ids)
        
        logits = self.classifier(vec)
        logits = self.dropout(logits)
        prob = torch.sigmoid(logits)
        
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(prob[:,1], labels.float())
            return loss, prob
        else:
            return prob
        
        
        
def model_setting_codet5(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    pretrain_model = "Salesforce/codet5-small"
    config = T5Config.from_pretrained(pretrain_model)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model)
    encoder = T5ForConditionalGeneration.from_pretrained(pretrain_model, config=config)   
    
    model = CodeT5(encoder, config, tokenizer)

    model.to(args.device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(args.p))
    model.to(args.device) 
    model.eval()
            
    return model, tokenizer
        

class InputDataset(object):
    def __init__(self, code):
        self.code = code
            
class InputFeatures(object):
    def __init__(self, idx, source_ids):
        self.idx = idx
        self.source_ids = source_ids

class TextDataset():
    def __init__(self, code_list, tokenizer):
        self.dataset = []
        idx = 0
        self.block_size = 512
        for code in code_list:
            source_ids = tokenizer.encode(code, max_length=self.block_size, padding='max_length', truncation=True)
            input_feature = InputFeatures(idx, source_ids)
            self.dataset.append(input_feature)
            idx += 1  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i): 
        return torch.tensor(self.dataset[i].source_ids)
    

def detect_codet5(snippet_dir, model, tokenizer, args):
    code_list = []
    for fn in os.listdir(snippet_dir):
        fp = os.path.join(snippet_dir, fn)
        with open(fp, 'r') as f:
            code = f.read()
        code_list.append(code)
    detect_dataset = TextDataset(code_list, tokenizer)
    detect_loader = DataLoader(detect_dataset, num_workers=4, batch_size=64, pin_memory=True)
    
    probs = []
    for data in detect_loader:
        inputs = data.to(args.device)
    
        with torch.no_grad():
            prob = model(inputs, labels=None)
            probs.append(prob.cpu().numpy())
    
        probs = np.concatenate(probs, 0)
        preds = probs.argmax(-1)

        return preds
    