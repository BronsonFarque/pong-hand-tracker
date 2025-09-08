from Prepare_Data import *
from Activations import ReLu, Softmax, Tanh, Sigmoid, MaxPooling
import numpy as np


class LinearLayer:

    def __init__(self, input_size, output_size, bias=False, activation_func=None):

        self.input_size = input_size
        self.output_size = output_size
        self.bias = bias
        self.activation_func = activation_func()
        self.current_inputs = None
        self.pre_activated_outputs = None
        self.bias_gradient = None
        self.update_matrix = None
        self.weights = np.random.randn(self.input_size, self.output_size) * np.sqrt(1 / self.input_size)

        if bias:
            self.weights = np.vstack(
                [self.weights, np.random.randn(1, self.output_size) * np.sqrt(1 / self.input_size)])

    def __call__(self, layer_inputs):

        if self.bias:
            layer_inputs = np.concatenate([layer_inputs, np.ones((layer_inputs.shape[0], 1))], axis=1)

        self.current_inputs = np.copy(layer_inputs)
        layer_output = layer_inputs @ self.weights

        if self.activation_func is not None:
            self.pre_activated_outputs = np.copy(layer_output)
            layer_output = self.activation_func(layer_output)

        return layer_output

    def back(self, loss_gradient):

        if self.activation_func is not None:
            loss_gradient = self.activation_func.derivative(self.pre_activated_outputs, loss_gradient)

        self.update_matrix = self.current_inputs.T @ loss_gradient
        new_loss_gradient = loss_gradient @ self.weights.T

        if self.bias:
            self.bias_gradient = np.sum(loss_gradient, axis=0, keepdims=True)
            new_loss_gradient = new_loss_gradient[:, :-1]

        return new_loss_gradient


class Convolution:

    def __init__(self, num_kernels, kernel_size=3, padding=1, stride=1, input_depth=1, dilation=1, activation_func=ReLu):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.input_depth = input_depth
        self.dilation = dilation
        self.kernels = np.random.randn(self.num_kernels, input_depth, kernel_size, kernel_size)

        self.activation_func = activation_func() if activation_func is not None else None
        self.pre_activated_output = None
        self.grad_kernels = None

    def pad_image(self, image):

        if self.padding > 0:
            return np.pad(image, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        return image

    def dilate_kernel(self, kernel):

        if self.dilation == 1:
            return kernel
        k_size = kernel.shape[-1]
        new_size = (k_size - 1) * self.dilation + 1
        dilated_kernel = np.zeros((new_size, new_size))
        for i in range(k_size):
            for j in range(k_size):
                dilated_kernel[i * self.dilation, j * self.dilation] = kernel[i, j]
        return dilated_kernel

    def __call__(self, layer_input, mode='conv'):

        if layer_input.shape[1] != self.input_depth:
            raise ValueError("Input depth does not match the kernel's input depth.")

        self.input_image = layer_input
        batch_size, depth, h, w = layer_input.shape
        layer_input = self.pad_image(layer_input)

        dilated_kernels = np.array([self.dilate_kernel(kernel) for kernel in self.kernels])
        dilated_k_size = dilated_kernels.shape[-1]
        out_h = ((h + 2 * self.padding - dilated_k_size) // self.stride) + 1
        out_w = ((w + 2 * self.padding - dilated_k_size) // self.stride) + 1
        pre_activated_output = np.zeros((batch_size, self.num_kernels, out_h, out_w))

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for i in range(out_h):
                    for j in range(out_w):
                        i_start, j_start = i * self.stride, j * self.stride
                        i_end, j_end = i_start + dilated_k_size, j_start + dilated_k_size
                        data_kernel = layer_input[b, :, i_start:i_end, j_start:j_end]
                        if data_kernel.shape == dilated_kernels[k].shape:
                            pre_activated_output[b, k, i, j] = np.sum(data_kernel * dilated_kernels[k])


        if mode == 'conv' and self.activation_func is not None:
            self.pre_activated_output = np.copy(pre_activated_output)
            layer_output = self.activation_func(pre_activated_output)
        else:
            layer_output = pre_activated_output

        return layer_output

    def back(self, loss_gradient, mode='conv'):

        batch_size, depth, h, w = self.input_image.shape
        grad_kernels = np.zeros_like(self.kernels)

        if mode == 'conv' and self.activation_func is not None:
            loss_gradient = self.activation_func.derivative(self.pre_activated_output, loss_gradient)

        layer_input = self.pad_image(self.input_image)

        for b in range(batch_size):
            for k in range(self.num_kernels):
                if mode == 'conv':
                    dilated_kernel = self.dilate_kernel(self.kernels[k])
                else:
                    dilated_kernel = self.kernels[k]
                for i in range(loss_gradient.shape[2]):
                    for j in range(loss_gradient.shape[3]):
                        i_start, j_start = i * self.stride, j * self.stride
                        i_end, j_end = i_start + dilated_kernel.shape[-1], j_start + dilated_kernel.shape[-1]
                        data_kernel = layer_input[b, :, i_start:i_end, j_start:j_end]
                        if data_kernel.shape == dilated_kernel.shape:
                            grad_kernels[k] += data_kernel * loss_gradient[b, k, i, j]

        self.grad_kernels = np.copy(grad_kernels)
        flipped_kernels = np.flip(self.kernels, axis=(2, 3))
        grad_input = np.zeros_like(self.input_image)

        for b in range(batch_size):
            for k in range(self.num_kernels):
                for i in range(h):
                    for j in range(w):
                        i_start, j_start = i * self.stride, j * self.stride
                        i_end, j_end = i_start + self.kernel_size, j_start + self.kernel_size
                        if i_end <= loss_gradient.shape[2] and j_end <= loss_gradient.shape[3]:
                            grad_input[b, :, i_start:i_end, j_start:j_end] += flipped_kernels[k] * loss_gradient[b, k, i, j]

        return grad_input, grad_kernels


class Flatten:

    def __call__(self, pre_flattened_output):

        self.input_shape = pre_flattened_output.shape
        flattened = (pre_flattened_output.reshape(pre_flattened_output.shape[0], -1))
        flattened /= np.sqrt(flattened.shape[1])
        return flattened

    def back(self, loss_gradient):

        if not hasattr(self, 'input_shape'):
            raise ValueError("Forward pass must be called before backpropagation.")
        reshaped_gradient = loss_gradient.reshape(self.input_shape)
        return reshaped_gradient


class LayerList:
    def __init__(self, *layers):

        self.model = list(layers)

    def append(self, *layers):

        for layer in layers:
            self.model.append(layer)

    def __iter__(self):

        return iter(self.model)

    def __getitem__(self, index):

        return self.model[index]

    def __call__(self, batch_input):

        """
        batch_input: np.ndarray, shape (B, C, H, W)
        returns:
            loc_preds:  np.ndarray, shape (B, A, 4)
            conf_preds: np.ndarray, shape (B, A, n_classes)
        """
        batch_locs, batch_confs = [], []
        for img in batch_input:
            x = img[np.newaxis, ...]
            feature_map = x
            locs, confs = [], []

            for layer in self.model:
                if isinstance(layer, SSDHead):
                    B, C, H, W = feature_map.shape
                    loc, conf = layer(feature_map)
                    locs.append(loc)
                    confs.append(conf)
                else:
                    feature_map = layer(feature_map)

            loc_preds_img = np.concatenate(locs, axis=1)
            conf_preds_img = np.concatenate(confs, axis=1)

            batch_locs.append(loc_preds_img[0])
            batch_confs.append(conf_preds_img[0])

        loc_preds = np.stack(batch_locs, axis=0)
        conf_preds = np.stack(batch_confs, axis=0)

        return loc_preds, conf_preds, []

    def backward(self, loss_gradient_tuple):

        cls_loss_grad = loss_gradient_tuple[0]  # shape (1, A, C)
        box_loss_grad = loss_gradient_tuple[1]  # shape (1, A, 4)

        for b in range(cls_loss_grad.shape[0]):
            loss_gradient = None
            box_grad = box_loss_grad[b:b + 1]
            cls_grad = cls_loss_grad[b:b + 1]

            for layer in reversed(self.model):

                if isinstance(layer, SSDHead):
                    conv_grad, box_grad, cls_grad = layer.back(box_grad, cls_grad)

                    if loss_gradient is None:
                        loss_gradient = conv_grad
                    else:
                        loss_gradient += conv_grad

                elif isinstance(layer, Convolution):
                    loss_gradient, _ = layer.back(loss_gradient)
                    loss_gradient = np.clip(loss_gradient, -1, 1)

                elif isinstance(layer, (MaxPooling, Flatten, LinearLayer)):
                    loss_gradient = layer.back(loss_gradient)
                    loss_gradient = np.clip(loss_gradient, -1, 1)

                else:
                    print(f"[LayerList.backward] Unrecognized layer type: {type(layer).__name__}")

    def update(self, optimizer, epoch):

        for layer in reversed(self.model):
            if isinstance(layer, SSDHead):
                optimizer.update(layer.loc_conv, epoch)
                optimizer.update(layer.conf_conv, epoch)
            else:
                optimizer.update(layer, epoch)


class SSDHead:

    def __init__(self, in_channels, num_priors, num_classes):

        self.saved_shapes = []
        self.loc_conv = Convolution(
            num_kernels=num_priors * 4,
            kernel_size=3,
            padding=1,
            stride=1,
            input_depth=in_channels,
            activation_func=None
        )

        self.conf_conv = Convolution(
            num_kernels=num_priors * num_classes,
            kernel_size=3,
            padding=1,
            stride=1,
            input_depth=in_channels,
            activation_func=None
        )
        self.num_priors = num_priors
        self.num_classes = num_classes
        self.saved_shape = None

    def __call__(self, feature_map):

        self.saved_shapes.clear()
        self.saved_shape = feature_map.shape  # (batch, C, H, W)
        batch, C, H, W = feature_map.shape

        loc = self.loc_conv(feature_map)
        conf = self.conf_conv(feature_map)

        loc = loc.reshape(batch, self.num_priors, 4,  H, W)
        loc = loc.transpose(0, 3, 4, 1, 2)
        loc = loc.reshape(batch, -1, 4)

        conf = conf.reshape(batch, self.num_priors, self.num_classes, H, W)
        conf = conf.transpose(0, 3, 4, 1, 2)
        conf = conf.reshape(batch, -1, self.num_classes)
        return loc, conf

    def back(self, grad_loc, grad_cls):

        batch, C, H, W = self.saved_shape
        expected_anchors = H * W * self.num_priors

        grad_loc_this = grad_loc[:, :expected_anchors, :]
        grad_cls_this = grad_cls[:, :expected_anchors, :]

        grad_loc_remain = grad_loc[:, expected_anchors:, :]
        grad_cls_remain = grad_cls[:, expected_anchors:, :]

        grad_loc_this = grad_loc_this.reshape(batch, H, W, self.num_priors, 4)
        grad_loc_this = grad_loc_this.transpose(0, 3, 4, 1, 2)
        grad_loc_this = grad_loc_this.reshape(batch, self.num_priors * 4, H, W)

        grad_cls_this = grad_cls_this.reshape(batch, H, W, self.num_priors, self.num_classes)
        grad_cls_this = grad_cls_this.transpose(0, 3, 4, 1, 2)
        grad_cls_this = grad_cls_this.reshape(batch, self.num_priors * self.num_classes, H, W)

        grad_input_loc, _ = self.loc_conv.back(grad_loc_this)
        grad_input_conf, _ = self.conf_conv.back(grad_cls_this)

        grad_input = grad_input_loc + grad_input_conf

        return grad_input, grad_loc_remain, grad_cls_remain
