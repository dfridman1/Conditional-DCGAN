import os
import argparse
import numpy as np
import warnings

import torch
import torch.optim as optim
from torchvision.datasets import CIFAR10, MNIST

from tqdm import tqdm

from cgan import helpers
from cgan.discriminator import Discriminator
from cgan.generator import Generator
from cgan.loss import vanilla_discriminator_criterion, vanilla_generator_criterion, \
    l2_discriminator_criterion, l2_generator_criterion


DATASET_NAME_TO_DATASET_BUILDER = {
    'mnist': MNIST,
    'cifar10': CIFAR10
}


def train(generator, discriminator, generator_criterion, discriminator_criterion, dataset, training_params):

    experiment_dirpath = training_params['experiment_dirpath']
    os.makedirs(experiment_dirpath, exist_ok=True)

    device, dtype = training_params['device'], torch.float32
    generator.to(device=device); discriminator.to(device=device)


    conditional = training_params.get('conditional', False)
    k = training_params.get('k', 1)
    batch_size = training_params.get('batch_size', 128)
    train_iters = training_params['train_iters']
    lr = training_params['lr']

    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    if training_params.get('lr_scheduler', False):
        gamma = helpers.get_exponential_lr_gamma(initial_lr=lr, final_lr=lr * 1e-2, num_iters=train_iters)
        d_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=d_optimizer, gamma=gamma)
        g_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=g_optimizer, gamma=gamma)
    else:
        d_lr_scheduler = g_lr_scheduler = None

    iterator = tqdm(range(1, train_iters + 1), total=train_iters)
    for it in iterator:
        generator.train(); discriminator.train()

        # discriminator
        d_loss = torch.tensor(0, dtype=dtype, device=device)
        d_optimizer.zero_grad()
        for _ in range(k):
            real_images_tensor, real_labels = helpers.sample_from_dataset(dataset, n=batch_size // 2)
            real_images_tensor = real_images_tensor.to(device=device, dtype=dtype)

            if conditional:
                real_labels_tensor = helpers.one_hot_labels_tensor(
                    real_labels, num_classes=len(dataset.classes)
                ).to(device=device, dtype=dtype)
                real_logits = discriminator(real_images_tensor, real_labels_tensor)
            else:
                real_logits = discriminator(real_images_tensor)

            z = helpers.generate_z(n=batch_size // 2, z_dim=generator.z_dim).to(device=device, dtype=dtype)

            if conditional:
                fake_labels = np.random.randint(0, len(dataset.classes), size=len(z))
                fake_labels_tensor = helpers.one_hot_labels_tensor(
                    fake_labels, num_classes=len(dataset.classes)
                ).to(device=device, dtype=dtype)
                fake_images_tensor = generator(z, fake_labels_tensor)
                fake_logits = discriminator(fake_images_tensor, fake_labels_tensor)
            else:
                fake_images_tensor = generator(z)
                fake_logits = discriminator(fake_images_tensor)

            d_loss += discriminator_criterion(real_logits=real_logits, fake_logits=fake_logits,
                                              label_smoothing=training_params.get('label_smoothing', False))
        d_loss /= k
        d_loss.backward()
        d_optimizer.step()

        # generator
        g_optimizer.zero_grad()

        z = helpers.generate_z(n=batch_size, z_dim=generator.z_dim).to(device=device, dtype=dtype)

        if conditional:
            fake_labels = np.random.randint(0, len(dataset.classes), size=len(z))
            fake_labels_tensor = helpers.one_hot_labels_tensor(
                fake_labels, num_classes=len(dataset.classes)
            ).to(device=device, dtype=dtype)
            fake_images_tensor = generator(z, fake_labels_tensor)
            fake_logits = discriminator(fake_images_tensor, fake_labels_tensor)
        else:
            fake_images_tensor = generator(z)
            fake_logits = discriminator(fake_images_tensor)

        g_loss = generator_criterion(fake_logits=fake_logits)
        g_loss.backward()
        g_optimizer.step()

        if g_lr_scheduler is not None:
            g_lr_scheduler.step(); d_lr_scheduler.step()

        iterator.set_description(
            f'D:{round(d_loss.cpu().item(), 5)}, G:{round(g_loss.cpu().item(), 5)}'
        )

        if it == 1 or it % training_params['show_every'] == 0:
            samples_per_class, num_classes = 10, len(dataset.classes)
            num_images = samples_per_class * num_classes
            z = helpers.generate_z(n=num_images, z_dim=generator.z_dim).to(device=device, dtype=dtype)
            if conditional:
                labels = np.tile(np.arange(num_classes), reps=samples_per_class)
                classnames = [dataset.classes[index] for index in labels]
                labels_tensor = helpers.one_hot_labels_tensor(labels, num_classes=num_classes).to(device=device, dtype=dtype)
                images = generator.generate(z, labels_tensor)
            else:
                classnames = None
                images = generator.generate(z)
            fig = helpers.show_images(images, classnames=classnames)
            iteration_str = str(it).zfill(len(str(train_iters)))
            out_filepath = os.path.join(experiment_dirpath, f'iteration_{iteration_str}.png')
            fig.savefig(out_filepath)
            fig.clf()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int)
    parser.add_argument('--experiment_dirpath', required=True)
    parser.add_argument('--dataset', choices=DATASET_NAME_TO_DATASET_BUILDER.keys(), required=True)
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--train_iters', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.add_argument('--show_every', type=int, default=1000)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--no_conditional', action='store_true')
    parser.add_argument('--l2_loss', action='store_true')
    parser.add_argument('--no_label_smoothing', action='store_true')
    parser.add_argument('--experiment_name', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_builder = DATASET_NAME_TO_DATASET_BUILDER[args.dataset]
    image_transform = lambda pil_image: helpers.pil_image_to_image_tensor(pil_image, image_size=args.image_size)
    dataset = helpers.concatenate_datasets([
        dataset_builder(root=args.dataset_dir, train=is_train, transform=image_transform, download=True)
        for is_train in (False, True)
    ])
    print(f'#images = {len(dataset)}')

    num_channels = dataset[0][0].size(0)
    image_size = dataset[0][0].size(1)

    num_additional_features = 0 if args.no_conditional else len(dataset.classes)
    generator = Generator(z_dim=args.z_dim, image_size=image_size, num_channels=num_channels, num_additional_features=num_additional_features)
    discriminator = Discriminator(image_size=image_size, num_channels=num_channels, num_additional_features=num_additional_features)

    if args.l2_loss:
        generator_criterion = l2_generator_criterion
        discriminator_criterion = l2_discriminator_criterion
    else:
        generator_criterion = vanilla_generator_criterion
        discriminator_criterion = vanilla_discriminator_criterion

    if torch.cuda.is_available():
        if args.gpu_id is not None:
            device = f'cuda:{args.gpu_id}'
        else:
            warnings.warn("--gpu_id not specified: running on CPU")
            device = 'cpu'
    else:
        device = 'cpu'

    training_params = {
        'device': device,
        'train_iters': args.train_iters,
        'lr': args.lr,
        'lr_scheduler': args.lr_scheduler,
        'show_every': args.show_every,
        'experiment_dirpath': os.path.join(args.experiment_dirpath, args.experiment_name),
        'k': args.k,
        'conditional': not args.no_conditional,
        'label_smoothing': not args.no_label_smoothing
    }

    train(generator=generator, discriminator=discriminator, generator_criterion=generator_criterion,
          discriminator_criterion=discriminator_criterion, dataset=dataset, training_params=training_params)


if __name__ == '__main__':
    main()
