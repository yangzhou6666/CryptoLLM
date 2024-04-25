python ./test.py \
    --do_test \
    --test_model=./saved_codegpt_25171106/checkpoint/codegpt.bin \
    --test_file=../dataset/test_dataset.jsonl \
    --output_dir=./test \
    --tokenizer_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --model_path=microsoft/CodeGPT-small-java-adaptedGPT2 \
    --num_labels 2 \
    --block_size 512 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --dropout_prob 0.5 \
    2>&1 | tee test.log