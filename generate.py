import torch
from torchvision.utils import save_image
import torch.nn as nn
import numpy as np

# Define your generator class (if not already imported)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
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


# Load the model
def load_model(model_path):
    generator = Generator(100, 3, 64)  # Initialize the generator
    generator.load_state_dict(torch.load(model_path))  # Load the state dict
    generator.eval()  # Set to evaluation mode
    return generator


# Generate an image
def generate_image(generator, output_path, latent_dim=100):
    # Create a random latent vector
    z = torch.randn(1, latent_dim, 1, 1)  # Batch size 1
    with torch.no_grad():
        generated_image = generator(z)  # Generate image
    # Scale from [-1, 1] to [0, 1] (if Tanh was used in the output)
    generated_image = (generated_image + 1) / 2
    # Save the image
    save_image(generated_image, output_path)


if __name__ == "__main__":
    model_path = "generator_22JAN.pt"  # Path to your saved model file
    output_path = "generated_image1.png"  # Path to save the generated image

    generator = load_model(model_path)
    generate_image(generator, output_path)
    print(f"Image saved to {output_path}")
