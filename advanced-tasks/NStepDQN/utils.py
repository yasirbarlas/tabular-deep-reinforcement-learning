import numpy as np

def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor

    optimizer: optimizer whose learning rate must be shrunk
    shrink_factor: factor in interval (0, 1) to multiply learning rate with
    """

    print("\nDECAYING learning rate")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]["lr"]))

def moving_average(data, window_size = 100):
    return np.convolve(a = data, v = np.ones(window_size), mode = "valid") / window_size