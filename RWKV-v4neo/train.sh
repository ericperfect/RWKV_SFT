python train.py \
    --load_model "" \
    --proj_dir "" \
    --wandb "rwkv" \
    --data_file "" \
    --my_data_file "" \
    --data_type binidx \
    --vocab_size 50277 \
    --ctx_len 8192 \
    --accumulate_grad_batches 1 \
    --epoch_steps 1000 \
    --epoch_count 10 \
    --epoch_begin 0 \
    --epoch_save 1 \
    --micro_bsz 8 \
    --n_layer 32 \
    --n_embd 4096 \
    --pre_ffn 0 \
    --head_qk 0 \
    --lr_init 1e-4 \
    --lr_final 5e-6 \
    --warmup_steps 50 \
    --beta1 0.9 \
    --beta2 0.999 \
    --adam_eps 1e-8 \
    --accelerator gpu \
    --devices 1 \
    --precision bf16 \
    --strategy deepspeed_stage_2 \
    --grad_cp 1 \
    --lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.01 \
    --lora_parts=att,ffn,time,ln \
    --my_qa_mask 1