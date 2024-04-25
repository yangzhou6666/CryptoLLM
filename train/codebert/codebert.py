from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import random
import json
import numpy as np
import torch
from tqdm import tqdm

from model import CodeBERT

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import (get_linear_schedule_with_warmup, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

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
    
    
class EarlyStopping(object):
    def __init__(self, patience=3, args=None):
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)    
        
        checkpoint_prefix = 'codebert.bin'
        output_dir = os.path.join(checkpoint_dir, checkpoint_prefix)
          
        self._min_loss = np.inf
        self._patience = patience
        self._path = output_dir
        self.__counter = 0
 
    def early_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), self._path)
            logger.info("  *** Saving model checkpoint to %s", self._path)
            
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return "stop"
            return "stay"
        return "go"
    
    def counter(self):
        return self.__counter
    
       
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
                code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
                source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
                source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
                padding_length = args.block_size - len(source_ids)
                source_ids+=[tokenizer.pad_token_id]*padding_length
                input_feature = InputFeatures(source_tokens, source_ids, js['label'])
                self.dataset.append(input_feature)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):       
        return torch.tensor(self.dataset[i].input_ids),torch.tensor(self.dataset[i].label)


def train(args, model, tokenizer):
    train_dataset = TextDataset(tokenizer, args, args.train_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    logger.info(f"train_dataset length: {len(train_dataset)}")

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps*0.1, num_training_steps=max_steps)
    
    model.zero_grad()
    
    early_stopper = EarlyStopping(patience=3, args=args)
    
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

            loss.backward()
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description("[Train] epoch {} loss {}".format(idx+1, round(np.mean(losses),3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        results, labels, preds, eval_dataset = valid(args, model, tokenizer, idx)
        logger.info("=======================================")
        logger.info("\n"+classification_report(labels, preds, zero_division=0))
        logger.info("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(results['acc'],results['pre'],results['rec'],results['f1']))
        logger.info("TPR: {:.4f}, TNR: {:.4f}, FPR: {:.4f}, FNR: {:.4f}".format(results['TPR'],results['TNR'],results['FPR'],results['FNR']))
        logger.info("loss: {:.4f}, perplexity: {:.4f}".format(results['loss'],results['perplexity']))
        
        stop = early_stopper.early_stop(model, results['loss'])
        
        if stop == "stop":
            logger.info(f"EarlyStopping STOP: epoch{(idx+1) - early_stopper.counter()}")
            logger.info(f"END!")
            logger.info("======================================================================================================================================")
            break
        elif stop == "go":
            logger.info("  *** Loss = %s", round(results['loss'],4))
            
            with open(os.path.join(args.output_dir, f"predic.txt"),'w') as f:
                for example, pred in zip(eval_dataset.dataset, preds):
                    f.write(str(pred)+'\n')

            with open(os.path.join(args.output_dir, f"result.txt"),'w') as f:
                f.write(f"epoch{idx+1}\n")
                f.write("loss: {:.4f}, perplexity: {:.4f}\n".format(results['loss'], results['perplexity']))
                f.write("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}\n".format(results['acc'],results['pre'],results['rec'],results['f1']))
                f.write("TPR: {:.4f}, TNR: {:.4f}, FPR: {:.4f}, FNR: {:.4f}\n".format(results['TPR'],results['TNR'],results['FPR'],results['FNR']))
                f.write(classification_report(labels, preds, zero_division=0))
        else:
            logger.info(f"EarlyStopping STAY: epoch{(idx+1)} -> {early_stopper.counter()}")
            
        with open(os.path.join(args.output_dir, f"summary.txt"),'a') as f:
            f.write(f"epoch{idx+1}\n")
            f.write("loss: {:.4f}, perplexity: {:.4f}\n".format(results['loss'], results['perplexity']))
            f.write("acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}\n".format(results['acc'],results['pre'],results['rec'],results['f1']))
            f.write("TPR: {:.4f}, TNR: {:.4f}, FPR: {:.4f}, FNR: {:.4f}\n".format(results['TPR'],results['TNR'],results['FPR'],results['FNR']))
            f.write(classification_report(labels, preds, zero_division=0))
            f.write("===========================================================================================")
        logger.info("=======================================")                



def valid(args, model, tokenizer, idx):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
        
    eval_dataset = TextDataset(tokenizer, args, args.valid_file)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    logger.info(f"eval_dataset length: {len(eval_dataset)}")
    
    eval_loss = 0.0
    nb_eval_steps = 0
    logits = [] 
    labels = []
    
    model.eval()

    for batch in tqdm(eval_dataloader, desc=f"[Valid] epoch {idx+1}"):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    
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
            
    results = {
        "loss": eval_loss, "perplexity": float(perplexity),
        "acc": round(acc,4), "pre": round(pre,4), "rec": round(rec,4), "f1": round(f1,4),
        "TPR": round(TPR,4), "TNR": round(TNR,4), "FPR": round(FPR,4), "FNR": round(FNR,4)
    }
    return results, labels, preds, eval_dataset
 
    
                        
                        
def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", action='store_true', required=True, help="Whether to run training.")

    parser.add_argument("--train_file", default=None, type=str, required=True, help="The input training data file (a text file).")
    parser.add_argument("--valid_file", default=None, type=str, required=True, help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path", default=None, type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_path", default="", type=str, help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument('--num_labels', type=int, default=2, help="num_labels")
    
    parser.add_argument("--dropout_prob", default=0.5, type=float, help="Dropout prob.") 
    parser.add_argument("--block_size", default=-1, type=int, help="Optional input sequence length after tokenization.") 
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=2024, help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=30, help="num_train_epochs")

    args = parser.parse_args()
    
    now = datetime.datetime.now()
    args.output_dir = f"{args.output_dir}_{now.strftime('%d%H%M%S')}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.n_gpu = torch.cuda.device_count()
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    set_seed(args.seed)

    # microsoft/codebert-base
    config = RobertaConfig.from_pretrained(args.model_path)
    config.num_labels = args.num_labels
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    encoder = RobertaForSequenceClassification.from_pretrained(args.model_path, config=config)    

    model = CodeBERT(encoder, config, tokenizer, args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    if args.do_train:
        logger.info("================================")
        train(args, model, tokenizer)


if __name__ == "__main__":
    main()