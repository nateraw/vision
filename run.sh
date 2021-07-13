python -i train.py \
    --output_dir outputs/ \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name nateraw/rock_paper_scissors \
    --num_train_epochs 4 \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
    --remove_unused_columns False \
    --do_train --do_eval \
    --overwrite_output_dir True \
    --fp16 True \
    --dataloader_num_workers 16 \
    --metric_for_best_model accuracy \
    --evaluation_strategy epoch