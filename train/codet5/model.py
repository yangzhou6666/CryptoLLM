import torch
import torch.nn as nn

    
class CodeT5(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.args = args

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
        source_ids = source_ids.view(-1, self.args.block_size)
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
      