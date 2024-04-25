python ./codet5.py \
    --do_train \
    --train_file=../dataset/train_dataset.jsonl \
    --valid_file=../dataset/valid_dataset.jsonl \
    --output_dir=./saved_codet5 \
    --tokenizer_path=Salesforce/codet5-small \
    --model_path=Salesforce/codet5-small \
    --epochs 1 \
    --num_labels 2 \
    --block_size 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --dropout_prob 0.5 \
    2>&1 | tee run.log