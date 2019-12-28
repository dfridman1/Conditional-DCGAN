import os
import argparse
import numpy as np

import torch
import torch.optim as optim
from torchvision.datasets import CIFAR10, MNIST

from tqdm import tqdm

from cgan import helpers
from cgan.discriminator import Discriminator
from cgan.generator import Generator
from cgan.loss import vanilla_discriminator_criterion, vanilla_generator_criterion


DATASET_NAME_TO_DATASET_BUILDER = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}


def train(generator, discriminator, generator_criterion, discriminator_criterion, dataset, training_params):

    experiment_dirpath = training_params['experiment_dirpath']
    os.makedirs(experiment_dirpath, exist_ok=True)

    device, dtype = training_params['device'], torch.float32
    generator.to(device=device); discriminator.to(device=device)

    g_optimizer = optim.Adam(generator.parameters(), lr=training_params.get('lr', 1e-3), betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=training_params.get('lr', 1e-3), betas=(0.5, 0.999))

    k = training_params.get('k', 1)
    batch_size = training_params.get('batch_size', 128)
    train_iters = training_params['train_iters']

    iterator = tqdm(range(1, train_iters + 1), total=train_iters)
    for it in iterator:
        generator.train(); discriminator.train()

        # discriminator
        d_loss = torch.tensor(0, dtype=dtype, device=device)
        d_optimizer.zero_grad()
        for _ in range(k):
            real_images_tensor, real_labels = helpers.sample_from_dataset(dataset, n=batch_size // 2)
            real_images_tensor = real_images_tensor.to(device=device, dtype=dtype)
            real_logits = discriminator(real_images_tensor)

            z = helpers.generate_z(n=batch_size // 2, z_dim=generator.z_dim).to(device=device, dtype=dtype)
            fake_images_tensor = generator(z)
            fake_labels = np.random.randint(0, len(dataset.classes), size=len(fake_images_tensor))
            fake_logits = discriminator(fake_images_tensor)

            d_loss += discriminator_criterion(real_logits=real_logits, fake_logits=fake_logits)
        d_loss /= k
        d_loss.backward()
        d_optimizer.step()

        # generator
        g_optimizer.zero_grad()

        z = helpers.generate_z(n=batch_size // 2, z_dim=generator.z_dim).to(device=device, dtype=dtype)
        fake_images_tensor = generator(z)
        fake_labels = np.random.randint(0, len(dataset.classes), size=len(fake_images_tensor))
        fake_logits = discriminator(fake_images_tensor)

        g_loss = generator_criterion(fake_logits=fake_logits)
        g_loss.backward()
        g_optimizer.step()

        iterator.set_description(
            f'D:{round(d_loss.cpu().item(), 5)}, G:{round(g_loss.cpu().item(), 5)}'
        )

        if it == 1 or it % training_params['show_every'] == 0:
            images = helpers.unnormalize(fake_images_tensor).cpu().detach().numpy()
            images = images.transpose(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            images = (images * 255).round().astype(np.uint8)
            fig = helpers.show_images(images)
            fig.suptitle(f'Iteration: {it}')
            out_filepath = os.path.join(experiment_dirpath, f'iteration_{it}.png')
            fig.savefig(out_filepath)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--experiment_dirpath', required=True)
    parser.add_argument('--dataset', choices=DATASET_NAME_TO_DATASET_BUILDER.keys(), required=True)
    parser.add_argument('--dataset_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
    parser.add_argument('--z_dim', type=float, default=128)
    parser.add_argument('--k', type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.dataset_dir)

    dataset_builder = DATASET_NAME_TO_DATASET_BUILDER[args.dataset]
    dataset = dataset_builder(root=args.dataset_dir, train=True, transform=helpers.pil_image_to_image_tensor)

    num_channels = dataset[0][0].size(0)
    image_size = dataset[0][0].size(1)

    generator = Generator(z_dim=args.z_dim, image_size=image_size, num_channels=num_channels)
    discriminator = Discriminator(image_size=image_size, num_channels=num_channels)

    generator_criterion = vanilla_generator_criterion
    discriminator_criterion = vanilla_discriminator_criterion

    training_params = {
        'device': torch.device(f'cuda:{args.gpu_id}'),
        'train_iters': 1000,
        'show_every': 100,
        'experiment_dirpath': args.experiment_dirpath,
        'k': args.k
    }

    train(generator=generator, discriminator=discriminator, generator_criterion=generator_criterion,
          discriminator_criterion=discriminator_criterion, dataset=dataset, training_params=training_params)


if __name__ == '__main__':
    main()
