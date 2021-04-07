#!usr/bin/python

"""
CAM visualization
"""

import math
import argparse
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image

from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM#, ISCAM
from torchcam.utils import overlay_mask

DENSENET_CONFIG = {_densenet: dict(conv_layer='features', fc_layer='classifier')
                   for _densenet in models.densenet.__dict__.keys()}

MODEL_CONFIG = {**DENSENET_CONFIG}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Pretrained imagenet model
model = models.__dict__['densenet121'](pretrained=True).to(device=device)
conv_layer = MODEL_CONFIG['densenet121']['conv_layer']
fc_layer = MODEL_CONFIG['densenet121']['fc_layer']

# Image
img = 'https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
img_path = BytesIO(requests.get(img).content)
pil_img = Image.open(img_path, mode='r').convert('RGB')

# Preprocess image
img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=device)

# Hook the corresponding layer in the model
cam_extractors = [CAM(model, conv_layer, fc_layer)]

extractor = cam_extractors[0]

# Don't trigger all hooks
extractor._hooks_enabled = False

idx = 0

extractor._hooks_enabled = True

model.zero_grad()
scores = model(img_tensor.unsqueeze(0))

# Select the class index
class_idx = 232 #scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

# Use the hooked data to compute activation map
activation_map = extractor(class_idx, scores).cpu()

# Clean data
extractor.clear_hooks()
extractor._hooks_enabled = False
# Convert it to PIL image
# The indexing below means first image in batch
heatmap = to_pil_image(activation_map, mode='F')
# Plot the result
result = overlay_mask(pil_img, heatmap)

plt.imshow(result)
# extractor.__class__.__name__

plt.tight_layout()
plt.show()
