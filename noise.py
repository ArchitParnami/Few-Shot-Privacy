import torch
from torch.nn.functional import avg_pool2d
import numpy as np
from PIL import Image, ImageFilter

def pixelate(img, b):
    expand = False
    if len(img.shape) == 2:
        expand = True
        img = np.expand_dims(img, axis=2)
    img = torch.from_numpy(img).unsqueeze(0).permute(0,3,1,2)
    px = avg_pool2d(img, (b, b))
    px = px.permute(0,2,3,1).squeeze(0).numpy()
    if expand:
        px = np.squeeze(px)
    return px
    
def add_gaussian_noise(img, mean, stdev, noise_factor=1):
    noise = noise_factor * np.random.normal(mean, stdev, img.shape)
    noisy_image = np.clip(img + noise, 0, 255)
    return noisy_image

def add_laplace_noise(img, loc, scale, noise_factor=1):
    noise = noise_factor *  np.random.laplace(loc, scale, img.shape)
    noisy_image = np.clip(img + noise, 0, 255)
    return noisy_image

def blur(img, r):
    img = img.astype(np.uint8)
    I = Image.fromarray(img)
    I = I.filter(ImageFilter.GaussianBlur(radius=r))
    img = np.asarray(I, dtype=np.float32)
    return img

def dp_pix(img, b, m, eps):
    img = pixelate(img, b)
    num_channels = img.shape[2] if len(img.shape) > 2 else 1
    scale = (255 * m * num_channels) / (b * b * eps) 
    if num_channels > 1:
        img_noisy = np.zeros(img.shape)
        for i in range(num_channels):
            img_noisy[:,:,i] = add_laplace_noise(img[:,:,i], 0, scale)
    else:
        img_noisy = add_laplace_noise(img, 0, scale)
    return img_noisy

