#!/usr/bin/env python
# coding=utf-8

import torch
import numpy as np
from Buzznauts.utils import set_device


class OLS_pytorch(object):
    def __init__(self, device=None):
        if device is None:
            device = set_device()
        self.device = device
        self.coefficients = []
        self.X = None
        self.y = None

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = self._reshape_x(X)
        if len(y.shape) == 1:
            y = self._reshape_x(y)

        X = self._concatenate_ones(X)

        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        XtX = torch.matmul(X.t(), X)
        Xty = torch.matmul(X.t(), y.unsqueeze(2))
        XtX = XtX.unsqueeze(0)
        XtX = torch.repeat_interleave(XtX, y.shape[0], dim=0)
        #betas_cholesky, _ = torch.linalg.solve(Xty, XtX)
        betas_cholesky = torch.linalg.solve(XtX, Xty)

        self.coefficients = betas_cholesky

    def predict(self, entry):
        if len(entry.shape) == 1:
            entry = self._reshape_x(entry)
        entry = self._concatenate_ones(entry)
        entry = torch.from_numpy(entry).float().to(self.device)
        prediction = torch.matmul(entry, self.coefficients)
        prediction = prediction.cpu().numpy()
        prediction = np.squeeze(prediction).T
        return prediction

    def _reshape_x(self, X):
        return X.reshape(-1, 1)

    def _concatenate_ones(self, X):
        ones = np.ones(shape=X.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, X), 1)
