from unittest import TestCase
import torch
import numpy as np
import pytorch_influence_functions.influence_functions.embeddings as emb

from utils.logistic_regression import (
    LogisticRegression
)


class TestIHVPGrad(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Generate a pytorch model and a dummy dataset:
        cls.n_features = 10
        cls.n_classes = 3
        cls.n_params = cls.n_classes * cls.n_features + cls.n_features
        cls.wd = wd = 0.1
        cls.model = LogisticRegression(cls.n_classes, cls.n_features, wd=cls.wd)

        # Fill the model with parameters fitting to the dataset (computed beforehand)
        coef = [[-0.114, -0.251, 0.103, -0.299, -0.294, 0.010, -0.182, 0.474, -0.688, -0.414],
                [0.283, 0.149, -0.064, -0.052, 0.363, 0.048, -0.183, -0.337, 0.277, -0.086],
                [-0.169, 0.102, -0.039, 0.351, -0.069, -0.058, 0.365, -0.136, 0.411, 0.501]]
        intercept = [0.105, 0.197, -0.302]
        with torch.no_grad():
            cls.model.linear.weight = torch.nn.Parameter(
                torch.tensor(coef, dtype=torch.float)
            )
            cls.model.linear.bias = torch.nn.Parameter(
                torch.tensor(intercept, dtype=torch.float)
            )

        cls.gpu = 1 if torch.cuda.is_available() else -1
        if cls.gpu == 1:
            cls.model = cls.model.cuda()

    class CreateData(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            out_data = self.data[idx]

            return out_data

    def test_get_embeds_single(self):
        x, _ = self.model.test_set[0]
        target_subset = self.CreateData([x])
        target_loader = torch.utils.data.DataLoader(target_subset, batch_size=1, shuffle=False)

        embed = emb.get_embeds(self.model, target_loader, "linear")

        true = np.array([[1.14726, -0.74372, -0.40357]])
        assert np.allclose(embed, true, rtol=1e-4)

    def test_get_embeds_multi(self):
        x, _ = self.model.test_set[[0, 1, 2, 3]]
        target_subset = self.CreateData(x)
        target_loader = torch.utils.data.DataLoader(target_subset, batch_size=2, shuffle=False)

        embed = emb.get_embeds(self.model, target_loader, "linear")

        assert embed.shape == (4, 3)
