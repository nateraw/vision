```
python run_image_classification.py \
    --output_dir beans-vit/ \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --dataset_name nateraw/beans \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate 2e-4 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --fp16 True \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
    --metric_for_best_model accuracy \
    --num_train_epochs 4 \
    --evaluation_strategy epoch \
    --logging_strategy steps --logging_steps 20 \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337
```