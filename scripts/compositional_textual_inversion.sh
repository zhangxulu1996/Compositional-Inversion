export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/reference_images/cat"


accelerate launch --main_process_port=29501 \
  --gpu_ids=0 \
  train_compositional_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cute-cat>" --initializer_token="cat" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=5000 \
  --learning_rate=0.005 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_vectors=3 \
  --output_dir="./snapshot/compositional_textual_inversion/cat" \
  --anchor_loss_weight=0.95 \
  --anchor_loss="l2" \
  --recons_loss_weight=0.05 \
  --seed=8888