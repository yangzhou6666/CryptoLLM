python ./test.py \
    --do_test \
    --test_model=./saved_codebert_25163630/checkpoint/codebert.bin \
    --test_file=../dataset/test_dataset.jsonl \
    --output_dir=./test \
    --tokenizer_path=microsoft/codebert-base \
    --model_path=microsoft/codebert-base \
    --num_labels 2 \
    --block_size 512 \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --dropout_prob 0.5 \
    2>&1 | tee test.log