from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import random
import json
import numpy as np
import torch
from tqdm import tqdm

from model import CodeGPT

from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import (GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report



logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class InputFeatures(object):
    def __init__(self, input_tokens, input_ids, label,):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.dataset = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                code=' '.join(js['code'].split())
                code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
                source_tokens = code_tokens
                source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = args.block_size-len(source_ids)
                source_ids += [tokenizer.pad_token_id]*padding_length
                input_feature = InputFeatures(source_tokens, source_ids, js['label'])
                self.dataset.append(input_feature)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i): 
        return torch.tensor(self.dataset[i].input_ids),torch.tensor(self.dataset[i].label)


def test(args, model, tokenizer):
    test_dataset = TextDataset(tokenizer, args, args.test_file)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    
    logger.info("  Num examples = %d", len(test_dataset))
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.module.load_state_dict(torch.load(args.test_model))
    # model.load_state_dict(torch.load(args.test_model))
    model.to(args.device) 
    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0    
    logits, labels = [], []   
    
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        inputs = batch[0].to(args.device)        
        label = batch[1].to(args.device) 
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
                   
    logits = np.concatenate(logits, 0)
    labels=np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    TPR = TP/(TP+FN)*100
    TNR = TN/(TN+FP)*100
    FPR = FP/(FP+TN)*100
    FNR = FN/(TP+FN)*100
    
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, average='macro')
    rec = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    
    with open(os.path.join(args.output_dir, f"predic.txt"),'w') as f:
        for example, pred in zip(test_dataset.dataset, preds):
            f.write(str(pred)+'\n')
    
    with open(os.path.join(args.output_dir, f"result.txt"),'w') as f:        
        f.write("=======================================\n")
        f.write("loss: {:.4f}, perplexity: {:.4f}\n".format(eval_loss, perplexity))
        f.write("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}\n".format(acc, pre, rec, f1))
        f.write("TPR: {:.4f}, TNR: {:.4f}, FPR: {:.4f}, FNR: {:.4f}\n".format(TPR, TNR, FPR, FNR))
        f.write(classification_report(labels, preds, zero_division=0))




def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_test", action='store_true', help="Whether to run eval on the dev set.")  

    parser.add_argument("--test_file", default=None, type=str, required=True, help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_model", default=None, type=str, required=True, help="The model path to test.")
    
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path", default=None, type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_path", default="", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument('--num_labels', type=int, default=2, help="num_labels")
    
    parser.add_argument("--dropout_prob", default=0.5, type=float, help="Dropout prob.") 
    parser.add_argument("--block_size", default=-1, type=int, help="Optional input sequence length after tokenization.") 
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size per GPU/CPU.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=2024, help="random seed for initialization")
    
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    
    set_seed(args.seed)

    # microsoft/CodeGPT-small-java-adaptedGPT2
    config = GPT2Config.from_pretrained(args.model_path)
    config.num_labels = args.num_labels
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.pad_token = '<pad>'
    config.pad_token_id = 50261
    encoder = GPT2ForSequenceClassification.from_pretrained(args.model_path, config=config)
    
    model = CodeGPT(encoder, config, tokenizer, args)
    
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
            
    test(args, model, tokenizer)


if __name__ == "__main__":
    main()
             
    