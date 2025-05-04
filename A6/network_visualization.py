"""
Implements a network visualization in PyTorch.
Make sure to write device-agnostic code. For any function, initialize new tensors
on the same device as input tensors
"""

import torch
import torch.nn.functional as F


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from network_visualization.py!")


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    X.requires_grad_()

    logits = model(X)

    loss = F.cross_entropy(logits, y)

    loss.backward()

    saliency = torch.max(X.grad.data, dim=1)[0]

    return saliency


def make_adversarial_attack(X, target_y, model, max_iter=100, verbose=True):
    """
    Generate an adversarial attack that is close to X, but that the model classifies
    as target_y.

    Inputs:
    - X: Input image; Tensor of shape (1, 3, 224, 224)
    - target_y: An integer in the range [0, 1000)
    - model: A pretrained CNN
    - max_iter: Upper bound on number of iteration to perform
    - verbose: If True, it prints the pogress (you can use this flag for debugging)

    Returns:
    - X_adv: An image that is close to X, but that is classifed as target_y
    by the model.
    """
    # Initialize our adversarial attack to the input image, and make it require
    # gradient

    X_adv = X.clone()

    X_adv.requires_grad_()

    learning_rate = 0.1

    for i in range(max_iter):
        logits = model(X_adv)

        pred_class = torch.argmax(logits, dim=1).item()

        target_score = logits[:, target_y]
        max_score = torch.max(logits, dim=1)[0]

        if pred_class == target_y:
            break

        loss = logits[:, target_y].sum()

        model.zero_grad()

        if X_adv.grad is not None:
            X_adv.grad.zero_()

        loss.backward()

        grad = X_adv.grad.data
        grad_norm = torch.norm(grad, p=2)

        step = learning_rate * grad / (grad_norm + 1e-10)

        X_adv.data += step

        if verbose:
            print(
                f"Iteration {i + 1}: target score {target_score.item():.3f}, max score {max_score.item():.3f}"
            )

    return X_adv


def class_visualization_step(img, target_y, model, **kwargs):
    """
    Performs gradient step update to generate an image that maximizes the
    score of target_y under a pretrained model.

    Inputs:
    - img: random image with jittering as a PyTorch tensor
    - target_y: Integer in the range [0, 1000) giving the index of the class
    - model: A pretrained CNN that will be used to generate the image

    Keyword arguments:
    - l2_reg: Strength of L2 regularization on the image
    - learning_rate: How big of a step to take
    """

    l2_reg = kwargs.pop("l2_reg", 1e-3)
    learning_rate = kwargs.pop("learning_rate", 25)

    img.requires_grad_()

    logits = model(img)

    loss = logits[:, target_y].sum() - l2_reg * torch.norm(img, p=2)

    loss.backward()

    grad = img.grad.data
    grad_norm = torch.norm(grad, p=2)

    step = learning_rate * grad / (grad_norm + 1e-10)

    img.data += step

    model.zero_grad()
    img.grad.zero_()

    return img

    ########################################################################
    # TODO: Use the model to compute the gradient of the score for the     #
    # class target_y with respect to the pixels of the image, and make a   #
    # gradient step on the image using the learning rate. Don't forget the #
    # L2 regularization term!                                              #
    # Be very careful about the signs of elements in your code.            #
    # Hint: You have to perform inplace operations on img.data to update   #
    # the generated image using gradient ascent & reset img.grad to zero   #
    # after each step.                                                     #
    ########################################################################
    # Replace "pass" statement with your code
    pass
    ########################################################################
    #                             END OF YOUR CODE                         #
    ########################################################################
