import numpy as np
from Layers import LayerList, Convolution, LinearLayer, Flatten


class MeanSquaredError:

    def __call__(self, predicted, expected):

        return np.mean((predicted - expected) ** 2)

    def derivative(self, predicted, expected):

        num_samples = predicted.shape[0]
        return 2 * (predicted - expected) / num_samples


class CrossEntropyLoss:

    def __call__(self, predicted, expected):

        predicted = np.clip(predicted, 1e-12, 1 - 1e-12)
        loss = -np.sum(expected * np.log(predicted)) / predicted.shape[0]
        return loss

    def derivative(self, predicted, expected):

        return predicted - expected


class SmoothL1:

    def __init__(self, beta=1):

        self.beta = beta
        self.diff = None
        self.flag = None
        self.mask = None

    def __call__(self, predicted, expected, mask=None):

        diff = predicted - expected

        abs_diff = np.abs(diff)
        self.flag = abs_diff < self.beta

        loss = np.where(self.flag, (.5 * (diff**2)) / self.beta, abs_diff - .5 * self.beta)

        self.diff = diff
        self.mask = mask

        return np.sum(loss)

    def derivative(self, predicted, expected, mask=None):

        diff = predicted - expected
        flag = np.abs(diff) < self.beta
        grad = np.where(flag,
                        diff / self.beta,
                        np.sign(diff))
        if mask is not None:
            grad = grad * mask[..., None]
        return grad


class SGD:

    def __init__(self, alpha=0.01):

        self.alpha = alpha

    def update(self, layer):

        if isinstance(layer, Convolution):
            weight_gradients = layer.grad_kernels
            weight_gradients = np.clip(weight_gradients, -1, 1)
        elif isinstance(layer, LinearLayer):
            weight_gradients = layer.update_matrix
            weight_gradients = np.clip(weight_gradients, -1, 1)
        else:
            return

        if isinstance(layer, LinearLayer) and layer.bias:
            if layer.bias_gradient is not None:
                layer.bias -= self.alpha * layer.bias_gradient

        if isinstance(layer, Convolution):
            layer.kernels -= self.alpha * weight_gradients
        elif isinstance(layer, LinearLayer):
            layer.weights -= self.alpha * weight_gradients


class SGDWithMomentum:

    def __init__(self, alpha=0.01, beta=0.9, decay=.0005, decay_type="exponential"):

        self.alpha = alpha
        self.beta = beta
        self.velocity = {}
        self.decay = decay
        self.decay_type = decay_type

    def update(self, layer, epoch):

        if hasattr(layer, 'weights') or hasattr(layer, 'kernels'):
            if layer not in self.velocity:
                self.velocity[layer] = np.zeros_like(layer.weights) if isinstance(layer, LinearLayer) else np.zeros_like(layer.kernels)

        if isinstance(layer, Convolution):
            weight_gradients = layer.grad_kernels
            weight_gradients = np.clip(weight_gradients, -1, 1)
        elif isinstance(layer, LinearLayer):
            weight_gradients = layer.update_matrix
            weight_gradients = np.clip(weight_gradients, -1, 1)
        else:
            return

        self.velocity[layer] = self.beta * self.velocity[layer] + (1 - self.beta) * weight_gradients

        if isinstance(layer, LinearLayer):
            layer.weights -= self.alpha * self.velocity[layer]
        elif isinstance(layer, Convolution):
            layer.kernels -= self.alpha * self.velocity[layer]

        if isinstance(layer, LinearLayer) and layer.bias:
            if layer.bias_gradient is not None:
                layer.bias -= self.alpha * layer.bias_gradient

        self.alpha = self.apply_lr_decay(epoch)

    def apply_lr_decay(self, epoch):

        if self.decay_type == "exponential":
            return self.alpha * np.exp(-self.decay * epoch)
        elif self.decay_type == "step":
            return self.alpha * (0.1 ** (epoch // 10))  # Decay every 10 epochs
        elif self.decay_type == "inverse":
            return self.alpha / (1 + self.decay * epoch)
        else:
            return self.alpha
