import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    delta2 = delta * 2
    assert np.all(np.isclose(orig_x, x, tol)), \
        "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        # do (f(x+delta) - f(x-delta)( / 2*delta
        tmp_x = orig_x.copy()
        np.add.at(tmp_x, (ix), delta)
        tmp_fx = f(tmp_x)[0]
        np.subtract.at(tmp_x, (ix), delta2)

        numeric_grad_at_ix = (tmp_fx - f(tmp_x)[0]) / delta2

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different "
                  "at %s. Analytic: %2.5f, Numeric: %2.5f"
                  % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True
