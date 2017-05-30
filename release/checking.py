import numpy as np

def estimate_gradients(mlp, X, y, step=1e-5):
    """Get gradients of `mlp` on its loss

    Parameters
    ----------
    X : a numpy 2darray of shape (nb_sample, nb_feature)
    y : a numpy 1darray of shape (nb_sample)
    step : float which represents approximation for dx in finite difference approximation

    """
    layers = [mlp.dense1, mlp.dense2]
    dparams = []
    for layer in layers:
        for pname in ['W', 'b']:
            pvalue = getattr(layer, pname)
            h, dparam = np.zeros_like(pvalue), np.zeros_like(pvalue)
            it = np.nditer(pvalue, flags=['multi_index'])
            while not it.finished:
                ix = it.multi_index
                h[ix] = step
                setattr(layer, pname, pvalue-h)
                loss = mlp.forward(X, y)
                setattr(layer, pname, pvalue+h)
                dparam[ix] = (mlp.forward(X, y) - loss) / (2*step)
                setattr(layer, pname, pvalue)
                h[ix] = 0
                it.iternext()
            dparams.append(dparam)
    return dparams
