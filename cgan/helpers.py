import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset
from torchvision.transforms import Resize, ToTensor


def pil_image_to_image_tensor(pil_image, image_size=None):
    if image_size is not None:
        pil_image = Resize((image_size, image_size))(pil_image)
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


def one_hot_labels_tensor(labels, num_classes):
    one_hot = np.zeros(shape=(len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    return torch.from_numpy(one_hot)


def concatenate_datasets(datasets):
    assert len(datasets) > 0
    classes_lst = []
    for dataset in datasets:
        assert hasattr(dataset, 'classes') and isinstance(dataset.classes, list)
        classes_lst.append(tuple(dataset.classes))
    assert len(set(classes_lst)) == 1, "dataset must have the same classes"
    dataset = ConcatDataset(datasets)
    setattr(dataset, 'classes', datasets[0].classes)
    return dataset


def get_exponential_lr_gamma(initial_lr, final_lr, num_iters):
    return (final_lr / initial_lr) ** (1. / num_iters)


def show_images(images, classnames=None):
    assert images.ndim == 4 and images.shape[-1] in (1, 3)
    assert classnames is None or 0 < len(classnames) == len(images)
    cmap = None
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=3)
        cmap = 'gray'

    show_title = True
    if classnames is None:
        num_classes = int(np.ceil(np.sqrt(len(images))))
        classnames = np.tile(np.arange(num_classes), reps=len(images))[:len(images)]
        show_title = False

    classname_to_images = defaultdict(list)
    for classname, image in zip(classnames, images):
        classname_to_images[classname].append(image)


    num_classes = len(classname_to_images)
    samples_per_class = max(map(len, classname_to_images.values()))

    fig, axes = plt.subplots(nrows=samples_per_class, ncols=num_classes, figsize=(num_classes, samples_per_class))
    axes = [ax for ax_lst in axes for ax in ax_lst]

    used_axes = set()
    classname_to_images = sorted(classname_to_images.items(), key=lambda x: x[0])
    for i, (classname, images) in enumerate(classname_to_images):
        for j, image in enumerate(images):
            plt_idx = j * num_classes + i
            used_axes.add(plt_idx)
            ax = axes[plt_idx]
            ax.imshow(image, cmap=cmap)
            ax.axis('off')
            if show_title and j == 0:
                ax.set_title(classname, fontsize=10)
    for i, ax in enumerate(axes):
        if i not in used_axes:
            ax.remove()
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig
