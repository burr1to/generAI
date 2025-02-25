import torch
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # 64x64 output
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def load_model(model_path):
    generator = Generator(100, 3, 64)
    generator.load_state_dict(torch.load(model_path))
    generator.eval()
    return generator


def generate_image(generator, output_path, latent_dim=100):

    z = torch.randn(1, latent_dim, 1, 1)
    with torch.no_grad():
        generated_image = generator(z)
    generated_image = (generated_image + 1) / 2
    save_image(generated_image, output_path)


for i in range(1, 10):
    model_path = "generator_22JAN.pt"
    output_path = f"./generations/generated_image{i}.png"

    generator = load_model(model_path)
    generate_image(generator, output_path)
