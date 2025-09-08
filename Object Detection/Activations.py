import numpy as np


class ReLu:

    def __call__(self, pre_activation_output):

        return np.maximum(0, pre_activation_output)

    def derivative(self, pre_activated_output, grad_so_far):

        return np.where(pre_activated_output > 0, grad_so_far, 0)


class Tanh:

    def __call__(self, pre_activation_output):

        output = np.clip(pre_activation_output, 1000, -1000)
        return (np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output))

    def derivative(self, pre_activation_output, grad_so_far):

        output = np.clip(pre_activation_output, 1000, -1000)
        return (1 - ((np.exp(output) - np.exp(-output)) / (np.exp(output) + np.exp(-output)) ** 2)) * grad_so_far


class Sigmoid:

    def __call__(self, pre_activation_output):

        pre_activation_output = np.clip(pre_activation_output, -1000, 1000)
        return 1 / (1 + np.exp(-pre_activation_output))

    def derivative(self, pre_activated_output, grad_so_far):

        sig = 1 / (1 + np.exp(-np.clip(pre_activated_output, -1000, 1000)))
        return (sig * (1 - sig)) * grad_so_far


class Softmax:

    def __call__(self, pre_activation_output):

        pre_activation_output = np.clip(pre_activation_output, -10, 10)
        exp_shifted = np.exp(pre_activation_output - np.max(pre_activation_output, axis=1, keepdims=True))
        denominator = np.sum(exp_shifted, axis=1, keepdims=True)
        return exp_shifted / denominator

    def derivative(self, softmax_output, loss_grad):

        return loss_grad


class MaxPooling:

    def __init__(self, kernel_size=2, stride=2, padding=0):

        self.kernel_size = kernel_size
        self.stride = stride
        self.max_indices = None
        self.input_shape = None
        self.padding = padding


    def pad_image(self, image):

        if self.padding > 0:
            return np.pad(image, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                          mode='constant')

        return image

    def __call__(self, pre_pooled_output):

        if self.padding > 0:
            pad_image(pre_pooled_output)
        self.input_shape = pre_pooled_output.shape  # Store input shape
        batch_size, depth, height, width = pre_pooled_output.shape
        pooled_height = (height - self.kernel_size) // self.stride + 1
        pooled_width = (width - self.kernel_size) // self.stride + 1

        pooled_output = np.zeros((batch_size, depth, pooled_height, pooled_width))
        self.max_indices = np.zeros_like(pre_pooled_output, dtype=bool)

        for b in range(batch_size):
            for c in range(depth):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        patch = pre_pooled_output[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        pooled_output[b, c, i, j] = max_val
                        max_pos = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_indices[b, c, h_start + max_pos[0], w_start + max_pos[1]] = True

        return pooled_output

    def back(self, loss_gradient):

        gradients_wrt_input = np.zeros(self.input_shape, dtype=float)

        for b in range(loss_gradient.shape[0]):
            for c in range(loss_gradient.shape[1]):
                for i in range(loss_gradient.shape[2]):
                    for j in range(loss_gradient.shape[3]):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        gradients_wrt_input[b, c, h_start:h_end, w_start:w_end] += (
                                self.max_indices[b, c, h_start:h_end, w_start:w_end] * loss_gradient[b, c, i, j]
                            )
        return gradients_wrt_input
