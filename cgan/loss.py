import torch
import torch.nn as nn


def vanilla_generator_criterion(fake_logits):
    return nn.BCEWithLogitsLoss()(fake_logits, torch.ones_like(fake_logits))


def vanilla_discriminator_criterion(real_logits, fake_logits):
    loss = nn.BCEWithLogitsLoss()
    return loss(real_logits, torch.ones_like(real_logits)) + loss(fake_logits, torch.zeros_like(fake_logits))
