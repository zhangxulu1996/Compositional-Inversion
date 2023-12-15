import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel
import torchvision
import math

class CRITIC(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet-50 backbone:
        backbone = torchvision.models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(backbone.children())[:-2]
        self.visual = nn.Sequential(*modules)
        #self.visual.fc = nn.Linear(self.visual.fc.in_features, 2)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, x, src):
        # concat x and src
        batch = x.shape[0]
        x = torch.cat([x, src], dim=0)
        # visual features
        features = self.visual(x)
        # cosine similarity as logits
        # features = features.view(batch, 2, -1)
        gen_features = features[:batch].view(batch, -1)
        src_features = features[batch].view(1, -1)
        gen_features = gen_features / gen_features.norm(dim=-1, keepdim=True)
        src_features = src_features / src_features.norm(dim=-1, keepdim=True)
        # logit_scale = self.logit_scale.exp() #
        logits = gen_features @ src_features.t()
        return logits
    
    def inference(self, x):
        # visual features
        features = self.visual(x)
        # cosine similarity as logits
        features = features.view(features.size(0), -1)
        features = features / features.norm(dim=-1, keepdim=True)
        return features



class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation_fn = GELUActivation() # ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(4928, 8)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = hidden_states.flatten(1)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Layout_Predictor(nn.Module):
    def __init__(self, pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder.requires_grad_(False)
        self.mlp = MLP()
        
    def forward(self, input_ids):
        encoder_hidden_states = self.text_encoder(input_ids)[0]
        bbox = self.mlp(encoder_hidden_states)
        return bbox
        