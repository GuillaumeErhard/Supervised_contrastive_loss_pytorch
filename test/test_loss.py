import unittest
import torch
import numpy as np
import random

from loss.spc import SupervisedContrastiveLoss


class TestLoss(unittest.TestCase):
    def test_antipodes(self):
        for i in range(1000):
            c = random.randint(1, 256)
            temperature = round(random.uniform(0.01, 0.1), 2)

            a = torch.ones(1, c)
            b = -torch.ones(1, c)
            projections = torch.cat([a, a, b, b])
            targets = torch.tensor([0, 0, 1, 1], dtype=torch.int)

            spc_loss = SupervisedContrastiveLoss(temperature=temperature)
            loss_func_res = spc_loss(projections, targets).numpy()

            self.assertAlmostEqual(
                - np.log((np.exp(1 / temperature)) / ((np.exp(1 / temperature)) + (2 * np.exp(-1 / temperature)))),
                loss_func_res, places=2)