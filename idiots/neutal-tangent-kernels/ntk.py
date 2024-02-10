from neural_tangents import stax
from idiots.dataset.algorithmic import binary_op_loaders
import jax
import jax.numpy as jnp
import neural_tangents as nt
from jax import random



def create_ntk(layer_size=316, output_size=10):
    
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(layer_size),  
        stax.Relu(),
        stax.Dense(layer_size),
        stax.Relu(),
        stax.Dense(output_size)
    )
    
    return init_fn, apply_fn, kernel_fn

layer_size = 316
output_size = 10 

init_fn, apply_fn, kernel_fn = create_ntk(layer_size, output_size)


task: str = "x + y (mod 47)"
log_every: int = 100
eval_every: int = 1000
warmup_steps: int = 10
train_batch_size: int = 128
test_batch_size: int = 16
train_percentage: float = 0.3
weight_decay: float = 0.1
steps: int = int(1e5)

def loss_fn(predict_fn, ys, t, xs=None):
    mean, cov = predict_fn(t=t, get='ntk', x_test=xs, compute_cov=True)
    mean = jnp.reshape(mean, mean.shape[:1] + (-1,))
    var = jnp.diagonal(cov, axis1=1, axis2=2)
    ys = jnp.reshape(ys, (1, -1))

    mean_predictions = 0.5 * jnp.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2,
                                    axis=1)

    return mean_predictions

# predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=1e-4)

def predict_with_ntk(kernel_fn, x_train, y_train, x_test, y_test):
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=1e-4)
    y_pred, test_kernel = predict_fn(x_test, get='ntk', compute_cov=True)
    return y_pred, test_kernel

def main(_):
    dataset, train_loader, test_loader = binary_op_loaders(
        task, train_batch_size, test_batch_size, train_percentage
    )

    for x_train, y_train in train_loader:
        x_train = jnp.array(x_train.cpu().numpy())
        y_train = jnp.array(y_train.cpu().numpy())
        for x_test, y_test in test_loader:
            x_test = jnp.array(x_test.cpu().numpy())
            y_test = jnp.array(y_test.cpu().numpy())
            y_pred, test_kernel = predict_with_ntk(kernel_fn, x_train, y_train, x_test, y_test)
            print(y_pred, test_kernel)
            break

if __name__ == "__main__":
    main(0)