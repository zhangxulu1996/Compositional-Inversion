import clip
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, DetrForObjectDetection
from transformers import AutoImageProcessor

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)
        
        self.preprocess = clip_preprocess

        self.coco_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

    def detect_objects(self, image, prompt):
        if 'and' in prompt.split(' '):
            obj = prompt.split(' and ')[-1]
        elif 'in front of' in prompt:
            obj = prompt.split(' in front of ')[0]
        inputs = self.coco_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.coco_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]

        for label in results["labels"]:
            if self.detector.config.id2label[label.item()] == obj:
                return 1.
        return 0.

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, generated_images, src_images=None, src_img_features=None):
        if src_img_features is None and src_images is not None:
            src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, generated_images, text=None, text_features=None):
        if text_features is None and text is not None:
            text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()

