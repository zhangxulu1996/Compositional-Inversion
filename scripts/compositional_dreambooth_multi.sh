export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="./snapshot/compositional_dreambooth/cat_chair/"


accelerate launch --main_process_port=29503 \
  --gpu_ids=0 \
  train_compositional_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list='./data/concepts_list_train_multi.json' \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --modifier_token="<cute-cat>+<blue-chair>" \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --num_class_images=200 \
  --scale_lr --hflip  \
  --recons_loss_weight=0.6 \
  --anchor_loss='l2' \
  --anchor_loss_weight=0.4 \
  --seed=8888