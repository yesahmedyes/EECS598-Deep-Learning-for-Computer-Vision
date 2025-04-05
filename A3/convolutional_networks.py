"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

from functools import cache
import torch
import math
from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from convolutional_networks.py!")


class Conv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        pad, stride = conv_param["pad"], conv_param["stride"]

        N, _, H, W = x.shape

        F, _, HH, WW = w.shape

        h_prime = 1 + (H + 2 * pad - HH) // stride
        w_prime = 1 + (W + 2 * pad - WW) // stride

        out = torch.zeros(N, F, h_prime, w_prime, dtype=x.dtype, device=x.device)

        x_padded = torch.nn.functional.pad(
            x, (pad, pad, pad, pad), mode="constant", value=0
        )

        w_reshaped = w.reshape(F, -1)

        for i in range(h_prime):
            for j in range(w_prime):
                patch = x_padded[
                    :, :, i * stride : i * stride + HH, j * stride : j * stride + WW
                ]

                out[:, :, i, j] = patch.reshape(N, -1) @ w_reshaped.T + b

        return out, (x, w, b, conv_param)

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        x, w, _, conv_param = cache

        pad, stride = conv_param["pad"], conv_param["stride"]

        N, _, H, W = x.shape

        F, _, HH, WW = w.shape

        h_prime = 1 + (H + 2 * pad - HH) // stride
        w_prime = 1 + (W + 2 * pad - WW) // stride

        x_padded = torch.nn.functional.pad(
            x, (pad, pad, pad, pad), mode="constant", value=0
        )

        w_reshaped = w.reshape(F, -1)

        dx_padded = torch.zeros_like(x_padded)
        dw_reshaped = torch.zeros_like(w_reshaped)

        for i in range(h_prime):
            for j in range(w_prime):
                patch = x_padded[
                    :, :, i * stride : i * stride + HH, j * stride : j * stride + WW
                ]

                patch_reshaped = patch.reshape(N, -1)

                dpatch = (dout[:, :, i, j] @ w_reshaped).reshape(patch.shape)

                dx_padded[
                    :, :, i * stride : i * stride + HH, j * stride : j * stride + WW
                ] += dpatch

                dw_reshaped += dout[:, :, i, j].T @ patch_reshaped

        dx = dx_padded[:, :, pad : pad + H, pad : pad + W]
        dw = dw_reshaped.reshape(w.shape)
        db = dout.sum(dim=(0, 2, 3))

        return dx, dw, db


class MaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        N, C, H, W = x.shape

        pool_height, pool_width, stride = (
            pool_param["pool_height"],
            pool_param["pool_width"],
            pool_param["stride"],
        )

        h_prime = 1 + (H - pool_height) // stride
        w_prime = 1 + (W - pool_width) // stride

        out = torch.zeros((N, C, h_prime, w_prime), dtype=x.dtype, device=x.device)

        for i in range(h_prime):
            for j in range(w_prime):
                patch = x[
                    :,
                    :,
                    i * stride : i * stride + pool_height,
                    j * stride : j * stride + pool_width,
                ].reshape(N, C, -1)

                out[:, :, i, j] = torch.max(patch, dim=2).values

        cache = (x, pool_param)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        x, pool_param = cache

        N, C, H, W = x.shape

        pool_height = pool_param["pool_height"]
        pool_width = pool_param["pool_width"]
        stride = pool_param["stride"]

        h_prime = 1 + (H - pool_height) // stride
        w_prime = 1 + (W - pool_width) // stride

        dx = torch.zeros_like(x)

        for i in range(h_prime):
            for j in range(w_prime):
                patch = x[
                    :,
                    :,
                    i * stride : i * stride + pool_height,
                    j * stride : j * stride + pool_width,
                ]

                patch_reshape = patch.reshape(N, C, -1)

                _, idx = torch.max(patch_reshape, dim=2)

                mask = torch.zeros_like(patch_reshape)
                idx = idx.to(mask.device)
                mask.scatter_(2, idx.unsqueeze(2), 1)

                dx[
                    :,
                    :,
                    i * stride : i * stride + pool_height,
                    j * stride : j * stride + pool_width,
                ] += (mask * dout[:, :, i, j].unsqueeze(2)).reshape(patch.shape)

        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dims=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=torch.float64,
        device="cuda",
    ):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cuda' or 'cuda'
        """
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dims

        W1 = weight_scale * torch.randn(num_filters, C, filter_size, filter_size).to(
            device=device, dtype=dtype
        )

        b1 = torch.zeros(num_filters).to(device=device, dtype=dtype)

        W2 = weight_scale * torch.randn(num_filters * H // 2 * W // 2, hidden_dim).to(
            device=device, dtype=dtype
        )

        b2 = torch.zeros(hidden_dim).to(device=device, dtype=dtype)

        W3 = weight_scale * torch.randn(hidden_dim, num_classes).to(
            device=device, dtype=dtype
        )

        b3 = torch.zeros(num_classes).to(device=device, dtype=dtype)

        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "dtype": self.dtype,
            "params": self.params,
        }

        torch.save(checkpoint, path)

        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location="cuda")

        self.params = checkpoint["params"]
        self.dtype = checkpoint["dtype"]
        self.reg = checkpoint["reg"]

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]

        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        h1, cache1 = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        h2, cache2 = Linear_ReLU.forward(h1, W2, b2)
        scores, cache3 = Linear.forward(h2, W3, b3)

        if y is None:
            return scores

        grads = {}

        loss, dx = softmax_loss(scores, y)
        loss += self.reg * torch.sum(W1 * W1)
        loss += self.reg * torch.sum(W2 * W2)

        dh2, dw3, db3 = Linear.backward(dx, cache3)
        dh1, dw2, db2 = Linear_ReLU.backward(dh2, cache2)
        _, dw1, db1 = Conv_ReLU_Pool.backward(dh1, cache1)

        grads["W1"] = dw1 + 2 * self.reg * W1
        grads["b1"] = db1

        grads["W2"] = dw2 + 2 * self.reg * W2
        grads["b2"] = db2

        grads["W3"] = dw3 + 2 * self.reg * W3
        grads["b3"] = db3

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dims=(3, 32, 32),
        num_filters=[8, 8, 8, 8, 8],
        max_pools=[0, 1, 2, 3, 4],
        batchnorm=False,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        weight_initializer=None,
        dtype=torch.float,
        device="cuda",
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cuda' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters) + 1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == "cuda":
            device = "cuda:0"

        C, H, W = input_dims

        final_dim = 1
        h_prime, w_prime = H, W

        for i, filters in enumerate(num_filters):
            channels = C if i == 0 else num_filters[i - 1]

            if weight_scale == "kaiming":
                W = kaiming_initializer(
                    channels, filters, K=3, relu=True, device=device, dtype=dtype
                )
            else:
                W = weight_scale * torch.randn(filters, channels, 3, 3).to(
                    device=device, dtype=dtype
                )

            b = torch.zeros(filters, device=device, dtype=dtype)

            self.params[f"W{i + 1}"] = W
            self.params[f"b{i + 1}"] = b

            # Add gamma and beta parameters for batch normalization
            if self.batchnorm:
                gamma = torch.ones(filters, device=device, dtype=dtype)
                beta = torch.zeros(filters, device=device, dtype=dtype)

                self.params[f"gamma{i + 1}"] = gamma
                self.params[f"beta{i + 1}"] = beta

            if i in max_pools:
                h_prime //= 2
                w_prime //= 2

            final_dim = filters * h_prime * w_prime

        if weight_scale == "kaiming":
            W = kaiming_initializer(
                final_dim, num_classes, K=None, relu=False, device=device, dtype=dtype
            )
        else:
            W = weight_scale * torch.randn(final_dim, num_classes).to(
                device=device, dtype=dtype
            )

        b = torch.zeros(num_classes, device=device, dtype=dtype)

        self.params[f"W{self.num_layers}"] = W
        self.params[f"b{self.num_layers}"] = b

        self.bn_params = []

        if self.batchnorm:
            self.bn_params = [{"mode": "train"} for _ in range(len(num_filters))]

        if not self.batchnorm:
            params_per_macro_layer = 2
        else:
            params_per_macro_layer = 4

        num_params = params_per_macro_layer * len(num_filters) + 2

        msg = "self.params has the wrong number of elements. Got %d; expected %d"
        msg = msg % (len(self.params), num_params)

        assert len(self.params) == num_params, msg

        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)

            assert param.device == torch.device(device), msg

            msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)

            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "dtype": self.dtype,
            "params": self.params,
            "num_layers": self.num_layers,
            "max_pools": self.max_pools,
            "batchnorm": self.batchnorm,
            "bn_params": self.bn_params,
        }

        torch.save(checkpoint, path)

        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cuda")

        self.params = checkpoint["params"]
        self.dtype = dtype
        self.reg = checkpoint["reg"]
        self.num_layers = checkpoint["num_layers"]
        self.max_pools = checkpoint["max_pools"]
        self.batchnorm = checkpoint["batchnorm"]
        self.bn_params = checkpoint["bn_params"]

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)

        mode = "test" if y is None else "train"

        conv_param = {"stride": 1, "pad": 1}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param["mode"] = mode

        scores = X

        hidden_out = {}
        layer_caches = {}
        final_conv_shape = None

        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]

            if i == self.num_layers:
                N = scores.shape[0]

                final_conv_shape = scores.shape

                scores_flat = scores.reshape(N, -1)

                scores, layer_cache = Linear.forward(scores_flat, W, b)
            elif (i - 1) in self.max_pools:
                if self.batchnorm:
                    gamma = self.params[f"gamma{i}"]
                    beta = self.params[f"beta{i}"]
                    bn_param = self.bn_params[i - 1]

                    scores, layer_cache = Conv_BatchNorm_ReLU_Pool.forward(
                        scores, W, b, gamma, beta, conv_param, bn_param, pool_param
                    )
                else:
                    scores, layer_cache = Conv_ReLU_Pool.forward(
                        scores, W, b, conv_param, pool_param
                    )
            else:
                if self.batchnorm:
                    gamma = self.params[f"gamma{i}"]
                    beta = self.params[f"beta{i}"]
                    bn_param = self.bn_params[i - 1]

                    scores, layer_cache = Conv_BatchNorm_ReLU.forward(
                        scores, W, b, gamma, beta, conv_param, bn_param
                    )
                else:
                    scores, layer_cache = Conv_ReLU.forward(scores, W, b, conv_param)

            hidden_out[i] = scores
            layer_caches[i] = layer_cache

        if y is None:
            return scores

        grads = {}

        loss, dh = softmax_loss(scores, y)

        for i in range(self.num_layers, 0, -1):
            w = self.params[f"W{i}"]
            b = self.params[f"b{i}"]

            loss += self.reg * torch.sum(w * w)

            if i == self.num_layers:
                dh, dw, db = Linear.backward(dh, layer_caches[i])

                if final_conv_shape is not None and len(final_conv_shape) > 2:
                    dh = dh.reshape(final_conv_shape)

            elif (i - 1) in self.max_pools:
                if self.batchnorm:
                    dh, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(
                        dh, layer_caches[i]
                    )

                    grads[f"gamma{i}"] = dgamma
                    grads[f"beta{i}"] = dbeta
                else:
                    dh, dw, db = Conv_ReLU_Pool.backward(dh, layer_caches[i])

            else:
                if self.batchnorm:
                    dh, dw, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(
                        dh, layer_caches[i]
                    )

                    grads[f"gamma{i}"] = dgamma
                    grads[f"beta{i}"] = dbeta
                else:
                    dh, dw, db = Conv_ReLU.backward(dh, layer_caches[i])

            grads[f"W{i}"] = dw + 2 * self.reg * w
            grads[f"b{i}"] = db

        return loss, grads


def find_overfit_parameters():
    weight_scale = 1e-2
    learning_rate = 1e-2

    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    input_dims = data_dict["X_train"].shape[1:]

    model = DeepConvNet(
        input_dims=input_dims,
        num_classes=10,
        num_filters=[16, 16, 32, 32, 64, 64, 128, 128, 256],
        max_pools=[1, 3, 5, 7],
        weight_scale="kaiming",
        reg=5e-4,
        dtype=torch.float32,
        device="cuda",
    )

    data = {
        "X_train": data_dict["X_train"].to(dtype=dtype, device=device),
        "y_train": data_dict["y_train"].to(device=device),
        "X_val": data_dict["X_val"].to(dtype=dtype, device=device),
        "y_val": data_dict["y_val"].to(device=device),
    }

    solver = Solver(
        model,
        data,
        num_epochs=50,
        batch_size=128,
        update_rule=adam,
        optim_config={
            "learning_rate": 1e-3,
        },
        lr_decay=0.95,
        print_every=50,
        device="cuda",
    )

    return solver


def kaiming_initializer(
    Din, Dout, K=None, relu=True, device="cuda", dtype=torch.float32
):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2.0 if relu else 1.0

    weight = None

    if K is None:
        fan_in = Din

        std = math.sqrt(gain / fan_in)

        weight = std * torch.randn(Din, Dout, device=device, dtype=dtype)
    else:
        fan_in = Din * K * K

        std = math.sqrt(gain / fan_in)

        weight = std * torch.randn(Dout, Din, K, K, device=device, dtype=dtype)

    return weight


class BatchNorm(object):
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param["mode"]
        eps = bn_param.get("eps", 1e-5)
        momentum = bn_param.get("momentum", 0.9)

        N, D = x.shape

        running_mean = bn_param.get(
            "running_mean", torch.zeros(D, dtype=x.dtype, device=x.device)
        )

        running_var = bn_param.get(
            "running_var", torch.zeros(D, dtype=x.dtype, device=x.device)
        )

        out, cache = None, None

        if mode == "train":
            sample_mean = x.mean(dim=0)
            sample_var = x.var(dim=0, unbiased=False)

            x_mu = x - sample_mean
            std = torch.sqrt(sample_var + eps)
            out = x_mu / std

            out = gamma * out + beta

            cache = (x, sample_mean, sample_var, eps, gamma, beta)

            running_mean = momentum * running_mean + (1 - momentum) * sample_mean
            running_var = momentum * running_var + (1 - momentum) * sample_var

        elif mode == "test":
            out = (x - running_mean) / torch.sqrt(running_var + eps)

            out = gamma * out + beta

            cache = (x, running_mean, running_var, eps, gamma, beta)
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param["running_mean"] = running_mean.detach()
        bn_param["running_var"] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        x, sample_mean, sample_var, eps, gamma, beta = cache

        N, D = x.shape

        x_mu = x - sample_mean
        std = torch.sqrt(sample_var + eps)
        x_hat = x_mu / std

        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)

        dx_hat = dout * gamma

        dvar = (-0.5 * (dx_hat * x_mu).sum(dim=0)) / (std**3)

        dmu = (-dx_hat / std).sum(dim=0) + (dvar * (-2 / N) * x_mu).sum(dim=0)

        dx = dx_hat / std + dvar * (2 / N) * x_mu + dmu / N

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        x, sample_mean, sample_var, eps, gamma, beta = cache
        N, D = x.shape

        x_mu = x - sample_mean
        std = torch.sqrt(sample_var + eps)
        x_hat = x_mu / std

        dbeta = dout.sum(dim=0)
        dgamma = (dout * x_hat).sum(dim=0)

        dx_hat = dout * gamma

        dx = (
            (1.0 / N)
            * (1.0 / std)
            * (N * dx_hat - dx_hat.sum(dim=0) - x_hat * (dx_hat * x_hat).sum(dim=0))
        )

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        N, C, H, W = x.shape

        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)

        out_flat, cache = BatchNorm.forward(x_flat, gamma, beta, bn_param)

        out = out_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)

        spatial_cache = (cache, N, C, H, W)

        return out, spatial_cache

    @staticmethod
    def backward(dout, spatial_cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        cache, N, C, H, W = spatial_cache

        dout_flat = dout.permute(0, 2, 3, 1).reshape(-1, C)

        dx_flat, dgamma, dbeta = BatchNorm.backward(dout_flat, cache)

        dx = dx_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return dx, dgamma, dbeta


##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape

        stride, pad = conv_param["stride"], conv_param["pad"]

        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)

        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)

        tx = x.detach()
        tx.requires_grad = True

        out = layer(tx)

        cache = (x, w, b, conv_param, tx, out, layer)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache

            out.backward(dout)

            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()

            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = (
                torch.zeros_like(tx),
                torch.zeros_like(layer.weight),
                torch.zeros_like(layer.bias),
            )
        return dx, dw, db


class FastMaxPool(object):
    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape

        pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
        stride = pool_param["stride"]

        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)

        tx = x.detach()
        tx.requires_grad = True

        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache

            out.backward(dout)

            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):
    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)

        out, relu_cache = ReLU.forward(a)

        cache = (conv_cache, relu_cache)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache

        da = ReLU.backward(dout, relu_cache)

        dx, dw, db = FastConv.backward(da, conv_cache)

        return dx, dw, db


class Conv_ReLU_Pool(object):
    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)

        s, relu_cache = ReLU.forward(a)

        out, pool_cache = FastMaxPool.forward(s, pool_param)

        cache = (conv_cache, relu_cache, pool_cache)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache

        ds = FastMaxPool.backward(dout, pool_cache)

        da = ReLU.backward(ds, relu_cache)

        dx, dw, db = FastConv.backward(da, conv_cache)

        return dx, dw, db


class Linear_BatchNorm_ReLU(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)

        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)

        out, relu_cache = ReLU.forward(a_bn)

        cache = (fc_cache, bn_cache, relu_cache)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache

        da_bn = ReLU.backward(dout, relu_cache)

        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)

        dx, dw, db = Linear.backward(da, fc_cache)

        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)

        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)

        out, relu_cache = ReLU.forward(an)

        cache = (conv_cache, bn_cache, relu_cache)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache

        dan = ReLU.backward(dout, relu_cache)

        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)

        dx, dw, db = FastConv.backward(da, conv_cache)

        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):
    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)

        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)

        s, relu_cache = ReLU.forward(an)

        out, pool_cache = FastMaxPool.forward(s, pool_param)

        cache = (conv_cache, bn_cache, relu_cache, pool_cache)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache

        ds = FastMaxPool.backward(dout, pool_cache)

        dan = ReLU.backward(ds, relu_cache)

        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)

        dx, dw, db = FastConv.backward(da, conv_cache)

        return dx, dw, db, dgamma, dbeta
