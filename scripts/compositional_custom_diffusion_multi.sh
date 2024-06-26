export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="./snapshot/compositional_custom_diffusion/cat+dog/"


accelerate launch --main_process_port=29503 \
  --gpu_ids=0 \
  train_compositional_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list='./data/concepts_list_train_multi.json' \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_class_images=200 \
  --scale_lr --hflip  \
  --modifier_token "<cute-cat>+<black-dog>" \
  --recons_loss_weight=0.05 \
  --anchor_loss='l2' \
  --anchor_loss_weight=0.95 \
  --seed=8888