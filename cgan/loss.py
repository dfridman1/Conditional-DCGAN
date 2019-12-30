import torch
import torch.nn as nn
import torch.nn.functional as F


def vanilla_generator_criterion(fake_logits):
    return nn.BCEWithLogitsLoss()(fake_logits, torch.ones_like(fake_logits))


def vanilla_discriminator_criterion(real_logits, fake_logits, label_smoothing=False):
    loss = nn.BCEWithLogitsLoss()
    real_labels = torch.ones_like(real_logits)
    if label_smoothing:
        real_labels *= 0.9
    return loss(real_logits, real_labels) + loss(fake_logits, torch.zeros_like(fake_logits))


def l2_generator_criterion(fake_logits):
    fake_probabilities = F.sigmoid(fake_logits)
    fake_labels = torch.ones_like(fake_probabilities)
    return _square(fake_probabilities - fake_labels).mean()


def l2_discriminator_criterion(real_logits, fake_logits, label_smoothing=False):
    real_probabilities = F.sigmoid(real_logits)
    fake_probabilities = F.sigmoid(fake_logits)
    real_labels = torch.ones_like(real_probabilities)
    if label_smoothing:
        real_labels *= 0.9
    return _square(real_probabilities - real_labels).mean() + _square(fake_probabilities - torch.zeros_like(fake_probabilities)).mean()


def _square(x):
    return x * x
