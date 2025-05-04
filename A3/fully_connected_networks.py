"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
from a3_helper import softmax_loss
from eecs598 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from fully_connected_networks.py!")


class Linear(object):
    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """

        out = x.view(x.shape[0], -1) @ w + b

        cache = (x, w, b)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache

        dx = (dout @ w.T).view(x.shape)
        dw = x.view(x.shape[0], -1).T @ dout
        db = torch.sum(dout, dim=0)

        return dx, dw, db


class ReLU(object):
    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = torch.clamp(x, min=0)

        cache = x

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        x = cache

        dx = torch.where(x >= 0, dout, 0)

        return dx


class Linear_ReLU(object):
    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)

        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """

        fc_cache, relu_cache = cache

        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)

        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=torch.float64,
        device="cpu",
    ):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        w1 = torch.randn(input_dim, hidden_dim, dtype=dtype) * weight_scale
        b1 = torch.zeros(hidden_dim, dtype=dtype)

        w2 = torch.randn(hidden_dim, num_classes, dtype=dtype) * weight_scale
        b2 = torch.zeros(num_classes, dtype=dtype)

        self.params = {
            "W1": w1,
            "W2": w2,
            "b1": b1,
            "b2": b2,
        }

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "params": self.params,
        }

        torch.save(checkpoint, path)

        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")

        self.params = checkpoint["params"]
        self.reg = checkpoint["reg"]

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        h1, cache1 = Linear_ReLU.forward(X, W1, b1)
        scores, cache2 = Linear.forward(h1, W2, b2)

        if y is None:
            return scores

        grads = {}

        loss, dx = softmax_loss(scores, y)
        loss += self.reg * torch.sum(W1 * W1)
        loss += self.reg * torch.sum(W2 * W2)

        dh1, dw2, db2 = Linear.backward(dx, cache2)
        _, dw1, db1 = Linear_ReLU.backward(dh1, cache1)

        grads["W1"] = dw1 + 2 * self.reg * W1
        grads["b1"] = db1

        grads["W2"] = dw2 + 2 * self.reg * W2
        grads["b2"] = db2

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=0.0,
        reg=0.0,
        weight_scale=1e-2,
        seed=None,
        dtype=torch.float64,
        device="cpu",
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for i in range(self.num_layers):
            fan_in = input_dim if i == 0 else hidden_dims[i - 1]
            fan_out = num_classes if i == self.num_layers - 1 else hidden_dims[i]

            self.params[f"W{i + 1}"] = (
                torch.randn((fan_in, fan_out), dtype=dtype) * weight_scale
            )
            self.params[f"b{i + 1}"] = torch.zeros(fan_out, dtype=dtype)

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}

        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}

            if seed is not None:
                self.dropout_param["seed"] = seed

    def save(self, path):
        checkpoint = {
            "reg": self.reg,
            "dtype": self.dtype,
            "params": self.params,
            "num_layers": self.num_layers,
            "use_dropout": self.use_dropout,
            "dropout_param": self.dropout_param,
        }

        torch.save(checkpoint, path)

        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location="cpu")

        self.params = checkpoint["params"]
        self.dtype = dtype
        self.reg = checkpoint["reg"]
        self.num_layers = checkpoint["num_layers"]
        self.use_dropout = checkpoint["use_dropout"]
        self.dropout_param = checkpoint["dropout_param"]

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)

        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode

        scores = X

        hidden_out = {}
        layer_caches = {}

        for i in range(1, self.num_layers + 1):
            w = self.params[f"W{i}"]
            b = self.params[f"b{i}"]

            if i == self.num_layers:
                scores, layer_cache = Linear.forward(scores, w, b)
            else:
                scores, layer_cache = Linear_ReLU.forward(scores, w, b)

            hidden_out[i] = scores
            layer_caches[i] = layer_cache

        # If test mode return early
        if mode == "test":
            return scores

        grads = {}

        loss, dh = softmax_loss(scores, y)

        for i in range(self.num_layers, 0, -1):
            w = self.params[f"W{i}"]
            b = self.params[f"b{i}"]

            loss += self.reg * torch.sum(w * w)

            if i == self.num_layers:
                dh, dw, db = Linear.backward(dh, layer_caches[i])
            else:
                dh, dw, db = Linear_ReLU.backward(dh, layer_caches[i])

            grads[f"W{i}"] = dw + 2 * self.reg * w
            grads[f"b{i}"] = db

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)

    data = {
        "X_train": data_dict["X_train"].to(dtype=dtype, device=device),
        "y_train": data_dict["y_train"].to(device=device),  # keep labels as long
        "X_val": data_dict["X_val"].to(dtype=dtype, device=device),
        "y_val": data_dict["y_val"].to(device=device),  # keep labels as long
    }

    solver = Solver(
        model=model,
        data=data,
        update_rule=Solver.sgd,
        optim_config={"learning_rate": 1e-1},
        lr_decay=0.95,
        num_epochs=10,
        batch_size=100,
        print_every=100,
        device=device,
    )

    return solver


def get_three_layer_network_params():
    weight_scale = 0.05
    learning_rate = 0.3

    return weight_scale, learning_rate


def get_five_layer_network_params():
    weight_scale = 0.1
    learning_rate = 0.1

    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}

    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw

    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}

    lr = config.setdefault("learning_rate", 1e-2)
    mu = config.setdefault("momentum", 0.9)

    v = config.get("velocity", torch.zeros_like(w))

    v = mu * v - lr * dw

    next_w = w + v

    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}

    lr = config.setdefault("learning_rate", 1e-2)
    decay_rate = config.setdefault("decay_rate", 0.99)
    eps = config.setdefault("epsilon", 1e-8)
    cache = config.setdefault("cache", torch.zeros_like(w))

    cache = decay_rate * cache + (1 - decay_rate) * dw**2

    next_w = w - lr * dw / (torch.sqrt(cache) + eps)

    config["cache"] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}

    lr = config.setdefault("learning_rate", 1e-3)
    beta1 = config.setdefault("beta1", 0.9)
    beta2 = config.setdefault("beta2", 0.999)
    eps = config.setdefault("epsilon", 1e-8)
    m = config.setdefault("m", torch.zeros_like(w))
    v = config.setdefault("v", torch.zeros_like(w))
    t = config.setdefault("t", 0)

    t += 1

    m = beta1 * m + (1 - beta1) * dw
    mt = m / (1 - beta1**t)

    v = beta2 * v + (1 - beta2) * (dw**2)
    vt = v / (1 - beta2**t)

    next_w = w - lr * mt / (torch.sqrt(vt) + eps)

    config["m"] = m
    config["v"] = v
    config["t"] = t

    return next_w, config


class Dropout(object):
    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param["p"], dropout_param["mode"]

        if "seed" in dropout_param:
            torch.manual_seed(dropout_param["seed"])

        if mode == "train":
            mask = torch.rand_like(x) > p

            out = x * mask / (1 - p)
        elif mode == "test":
            out = x
            mask = None

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param["mode"]
        p = dropout_param["p"]

        dx = None

        if mode == "train":
            dx = dout * mask / (1 - p)
        elif mode == "test":
            dx = dout

        return dx
