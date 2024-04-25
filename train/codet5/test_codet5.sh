python ./test.py \
    --do_test \
    --test_model=./saved_codet5_25172812/checkpoint/codet5.bin \
    --test_file=../dataset/test_dataset.jsonl \
    --output_dir=./test \
    --tokenizer_path=Salesforce/codet5-small \
    --model_path=Salesforce/codet5-small \
    --num_labels 2 \
    --block_size 512 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --dropout_prob 0.5 \
    2>&1 | tee test.log