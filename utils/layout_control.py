import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from diffusers import StableDiffusionPipeline
from model import unet_2d_condition
import json
import os
from tqdm import tqdm
import math
from PIL import Image, ImageDraw, ImageFont
import logging


def compute_ca_loss(attn_maps_mid, attn_maps_up, bboxes, object_positions, device):
    loss = 0
    object_number = len(bboxes)
    if object_number == 0:
        return torch.tensor(0).float().to(device)
    for attn_map_integrated in attn_maps_mid:
        attn_map = attn_map_integrated.chunk(2)[1]

        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))
        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).to(device)
            for obj_box in bboxes[obj_idx]:

                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1)/ca_map_obj.reshape(b, -1).sum(dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss/len(object_positions[obj_idx]))

    for attn_map_integrated in attn_maps_up[0]:
        attn_map = attn_map_integrated.chunk(2)[1]
        #
        b, i, j = attn_map.shape
        H = W = int(math.sqrt(i))

        for obj_idx in range(object_number):
            obj_loss = 0
            mask = torch.zeros(size=(H, W)).to(device)
            for obj_box in bboxes[obj_idx]:
                x_min, y_min, x_max, y_max = int(obj_box[0] * W), \
                    int(obj_box[1] * H), int(obj_box[2] * W), int(obj_box[3] * H)
                mask[y_min: y_max, x_min: x_max] = 1

            for obj_position in object_positions[obj_idx]:
                ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
                # ca_map_obj = attn_map[:, :, object_positions[obj_position]].reshape(b, H, W)

                activation_value = (ca_map_obj * mask).reshape(b, -1).sum(dim=-1) / ca_map_obj.reshape(b, -1).sum(
                    dim=-1)

                obj_loss += torch.mean((1 - activation_value) ** 2)
            loss += (obj_loss / len(object_positions[obj_idx]))
    loss = loss / (object_number * (len(attn_maps_up[0]) + len(attn_maps_mid)))
    return loss

def Pharse2idx(prompt, phrases):
    phrases = [x.strip() for x in phrases.split(';')] # [cat, wooden_pot]
    prompt_list = prompt.replace(',', ' ,').strip('.').split(' ')
    object_positions = []
    for obj in phrases:
        obj_position = []
        for word in obj.split('_'):
            obj_first_index = prompt_list.index(word) + 1
            obj_position.append(obj_first_index)
        object_positions.append(obj_position)
    return object_positions


def setup_logger(save_path, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(os.path.join(save_path, f"{logger_name}.log"))
    file_handler.setLevel(logging.INFO)

    # Create a formatter to format log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the formatter for the file handler
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def inference(device, unet, vae, tokenizer, text_encoder, examples, num_steps, generator, guidance_scale=7.5):
    prompt, bboxes, identifier = examples['edited_prompt'], examples['bboxes'], examples['identifier']
    batch_size = 1
    noise_schedule = {'beta_start': 0.00085, 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'num_train_timesteps': 1000}
    inference = {'loss_scale': 30, 'loss_threshold': 0.2, 'max_iter': 5, 'max_index_step': 10}
    object_positions = Pharse2idx(prompt, identifier)
    # Encode Classifier Embeddings
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Encode Prompt
    input_ids = tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings]) # [2*77*768]

    latents = torch.randn(
        (batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    noise_scheduler = LMSDiscreteScheduler(beta_start=noise_schedule['beta_start'], beta_end=noise_schedule['beta_end'],
                                           beta_schedule=noise_schedule['beta_schedule'], num_train_timesteps=noise_schedule['num_train_timesteps'])

    noise_scheduler.set_timesteps(num_steps)

    latents = latents * noise_scheduler.init_noise_sigma

    loss = torch.tensor(10000)

    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        iteration = 0

        while loss.item() / inference['loss_scale'] > inference['loss_threshold'] and iteration < inference['max_iter'] and index < inference['max_index_step']:
            latents = latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=cond_embeddings)

            # update latents with guidance
            loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                                   object_positions=object_positions, device=device) * inference['loss_scale']

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents = latents - grad_cond * noise_scheduler.sigmas[index] ** 2   # update latents
            iteration += 1
            torch.cuda.empty_cache()

        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()

    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
    

class Layout_Control:
    def __call__(self, examples, num_steps=50, generator=None, guidance_scale=7.5):
        pil_images = inference(self.device, self.unet, self.vae, self.tokenizer, self.text_encoder, examples, num_steps=num_steps, generator=generator, guidance_scale=guidance_scale)
        return pil_images
    
    def load_textual_inversion(self, path):
        self.pipe.load_textual_inversion(path)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

    def load_attn_procs(self, dir, weight_name):
        self.unet.load_attn_procs(dir, weight_name=weight_name)
    
    def load_dreambooth_weights(self, dir):
        self.unet = None
        self.unet = unet_2d_condition.UNet2DConditionModel.from_pretrained(os.path.join(dir, 'unet')).to(self.pipe.device)
        if os.path.exists(os.path.join(dir, 'text_encoder')):
            self.text_encoder = None
            self.text_encoder = CLIPTextModel.from_pretrained(os.path.join(dir, 'text_encoder')).to(self.pipe.device)
        
    def __init__(self, model_id, device):
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
        self.unet = unet_2d_condition.UNet2DConditionModel(self.pipe.unet.config).from_pretrained(model_id, subfolder="unet").to(device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.device = device
        