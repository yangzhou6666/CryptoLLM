python ./codebert.py \
    --do_train \
    --train_file=../dataset/train_dataset.jsonl \
    --valid_file=../dataset/valid_dataset.jsonl \
    --output_dir=./saved_codebert \
    --tokenizer_path=microsoft/codebert-base \
    --model_path=microsoft/codebert-base \
    --epochs 30 \
    --num_labels 2 \
    --block_size 512 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --dropout_prob 0.5 \
    2>&1 | tee run.log