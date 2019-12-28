import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torchvision.transforms import ToTensor


def pil_image_to_image_tensor(pil_image):
    return normalize(ToTensor()(pil_image))


def normalize(image_tensor):
    return 2 * image_tensor - 1


def unnormalize(normalized_image_tensor):
    return (normalized_image_tensor + 1) / 2


def generate_z(n, z_dim):
    return torch.randn(n, z_dim)


def sample_from_dataset(dataset, n):
    assert n > 0
    indices = np.random.choice(len(dataset), size=n, replace=False)
    image_tensors, labels = zip(*[dataset[index] for index in indices])
    image_tensors = torch.cat([x.unsqueeze(0) for x in image_tensors])
    return image_tensors, labels


def show_images(images):
    assert images.ndim == 4
    num_channels = images.shape[-1]
    assert num_channels in (1, 3)
    images = np.reshape(images, [images.shape[0], -1, num_channels])
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    if num_channels == 3:
        out_shape = [sqrtimg, sqrtimg, num_channels]
        cmap = None
    else:
        out_shape = [sqrtimg, sqrtimg]
        cmap = 'gray'
    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(out_shape), cmap=cmap)
    return fig
