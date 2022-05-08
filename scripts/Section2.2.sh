#!bin/bash
python3 check_cuda.py
MODEL="splinter"
python3 -u train_sharc.py \
    --train_batch=16 \
    --gradient_accumulation_steps=2 \
    --epoch=5 \
    --seed=323 \
    --learning_rate=5e-5 \
    --loss_entail_weight=3.0 \
    --dsave="./out/{}" \
    --model=decision \
    --early_stop=dev_0a_combined \
    --data=./data/ \
    --data_type=decision_splinter_base \
    --prefix=train_decision \
    --trans_layer=2 \
    --eval_every_steps=300 \
    --pretrained_lm_path="tau/splinter-base"