import torch
import torch.nn as nn
import torch.nn.functional as F


def vanilla_generator_criterion(fake_logits):
    return nn.BCEWithLogitsLoss()(fake_logits, torch.ones_like(fake_logits))


def vanilla_discriminator_criterion(real_logits, fake_logits):
    loss = nn.BCEWithLogitsLoss()
    return loss(real_logits, torch.ones_like(real_logits)) + loss(fake_logits, torch.zeros_like(fake_logits))


def l2_generator_criterion(fake_logits):
    fake_probabilities = F.sigmoid(fake_logits)
    fake_labels = torch.ones_like(fake_probabilities)
    return _square(fake_probabilities - fake_labels).mean()


def l2_discriminator_criterion(real_logits, fake_logits):
    real_probabilities = F.sigmoid(real_logits)
    fake_probabilities = F.sigmoid(fake_logits)
    return _square(real_probabilities - torch.ones_like(real_probabilities)).mean() + _square(fake_probabilities - torch.zeros_like(fake_probabilities)).mean()


def _square(x):
    return x * x
