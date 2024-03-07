from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.grokking.training import restore as algorithmic_restore, restore_partial as algorithmic_restore_partial, eval_step
from idiots.experiments.classification.training import restore as mnist_restore, restore_partial as mnist_restore_partial
from idiots.utils import metrics
import neural_tangents as nt
from einops import rearrange
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
import json
import os
import warnings
import gc

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

warnings.filterwarnings('ignore')

# GP kernel object (for compatability with sklearn.gaussian_proccess)
class CustomKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn
        super().__init__()

    @jax.jit
    def __call__(self, X, Y=None):
        return self.kernel_fn(X, Y)
    
# Return the transformer state and training/test accuracy/loss for each timestep 
def eval_checkpoint(step, batch_size, ds_train, ds_test, num_classes, restore_manager, restore_partial_fn):
  config, state = restore_partial_fn(restore_manager, step, ds_train["x"][:1], num_classes)

  def eval_loss_acc(ds):
    for batch in DataLoader(ds, batch_size):
      logs = eval_step(state, batch, config.loss_variant)
      metrics.log(**logs)
    [losses, accuracies] = metrics.collect("eval_loss", "eval_accuracy")
    loss = jnp.concatenate(losses).mean().item()
    acc = jnp.concatenate(accuracies).mean().item()
    return loss, acc

  if len(ds_train['x']) > len(ds_test['x']): 
    ds_train = ds_train.select(range(len(ds_test['x'])))

  train_loss, train_acc = eval_loss_acc(ds_train)
  test_loss, test_acc = eval_loss_acc(ds_test)

  return state, train_loss, train_acc, test_loss, test_acc

"""
Returns a dataframe representing the checkpoint data of the *transformer*, containing: 
  - step (current checkpoint step)
  - state (of transformer network)
  - train_loss 
  - train_acc
  - test_loss 
  - test_acc 
"""
def extract_data_from_checkpoints(restore_manager, ds_train, ds_test, num_classes, total_steps, step_distance, restore_partial_fn, eval_checkpoint_batch_size=512):
  data = []
  for step in range(0, total_steps, step_distance):

    print(f"Loading Data: {(step // step_distance) + 1}/{total_steps // step_distance}")

    state, train_loss, train_acc, test_loss, test_acc = eval_checkpoint(step, eval_checkpoint_batch_size, ds_train, ds_test, num_classes, restore_manager, restore_partial_fn)
    data.append(
        {
            "step": step,
            "state": state,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )

  return pd.DataFrame(data)


# Parse the general checkpoint dataframe into useful sub-dataframes and lists 
def parse_general_checkpoint_dataframe(df):

    state_checkpoints = df['state'].tolist()

    df_loss = df[["step", "train_loss", "test_loss"]]
    df_loss = df_loss.melt("step", var_name="split", value_name="loss")
    df_loss["split"] = df_loss["split"].str.replace("_loss", "")

    df_acc = df[["step", "train_acc", "test_acc"]]
    df_acc = df_acc.melt("step", var_name="split", value_name="accuracy")
    df_acc["split"] = df_acc["split"].str.replace("_acc", "")

    steps = df["step"].tolist() 
    training_loss = df_loss[df_loss["split"] == "train"]["loss"].tolist()
    test_loss = df_loss[df_loss["split"] == "test"]["loss"].tolist()
    training_acc = df_acc[df_acc["split"] == "train"]["accuracy"].tolist()
    test_acc = df_acc[df_acc["split"] == "test"]["accuracy"].tolist()

    return state_checkpoints, steps, training_loss, test_loss, training_acc, test_acc 

# From X_test and Y_test, generate two (disjoint) datasets: one for calculating kernels and one for the remaining analysis (SVM & GP accuracy, and other metrics such as kernel alignment)
def generate_kernel_and_analysis_datasets(X_test, Y_test, num_kernel_samples, num_analysis_training_samples, num_analysis_test_samples, experiment_type):

  kernel_X = X_test[:num_kernel_samples]

  analysis_X_train = X_test[num_kernel_samples : num_kernel_samples+num_analysis_training_samples]
  analysis_Y_train = Y_test[num_kernel_samples : num_kernel_samples+num_analysis_training_samples]

  analysis_X_test = X_test[num_kernel_samples+num_analysis_training_samples : num_kernel_samples+num_analysis_training_samples+num_analysis_test_samples]
  analysis_Y_test = Y_test[num_kernel_samples+num_analysis_training_samples : num_kernel_samples+num_analysis_training_samples+num_analysis_test_samples]

  if experiment_type == "mnist": 
    analysis_X_train = rearrange(analysis_X_train, 'b h w -> b (h w)')
    analysis_X_test = rearrange(analysis_X_test, 'b h w -> b (h w)')

  return kernel_X, analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test

# Return a batched kernel function where trace_axes=() [for calculating DOTS]
@jax.jit
def compute_kernel_trace_axes_fn(transformer_state_apply_fn): 
  kernel_fn_trace_axes = nt.empirical_kernel_fn(transformer_state_apply_fn,
                       vmap_axes=0,
                       trace_axes=(),
                       implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)
  return kernel_fn_trace_axes

# Return a batched kernel function where trace_axes is not defined [for computing everything other than DOTS]
@jax.jit
def compute_kernel_fn(transformer_state_apply_fn):
  kernel_fn = nt.empirical_kernel_fn(transformer_state_apply_fn,
                     vmap_axes=0,
                     implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)
  return kernel_fn 

# Apply the kernel_trace_axes_fn to the values X with given transformer_state_params
@jax.jit(static_argnums=(2,3))
def compute_kernel_trace_axes(kernel_trace_axes_fn, transformer_state_params, X, batch_size):
  kernel_trace_axes_fn_batched = nt.batch(kernel_trace_axes_fn, device_count=-1, batch_size=batch_size)
  kernel_trace_axes = kernel_trace_axes_fn_batched(X, None, "ntk", transformer_state_params)
  kernel_trace_axes = rearrange(kernel_trace_axes, "b1 b2 d1 d2 -> (b1 d1) (b2 d2)")
  return kernel_trace_axes

# Apply the kernel_fn to the values X with given transformer_state_params 
@jax.jit(static_argnums=(2,3))
def compute_kernel(kernel_fn, transformer_state_params, X, batch_size):
  kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size)
  kernel = kernel_fn_batched(X, None, "ntk", transformer_state_params)
  return kernel 

# Compute DOTS on the kernel_trace_axes matrix 
@jax.jit
def compute_dots(kernel_trace_axes):
    kernel_rank = jax.jit(jnp.linalg.matrix_rank)(kernel_trace_axes)
    return kernel_rank.item()

# Create a custom_kernel_function for use in training the SVM and GP (mapping inputs X1 and X2 to a kernel matrix) 
def compute_custom_kernel_fn(kernel_fn, state_params):
  return lambda X1, X2: kernel_fn(X1, X2, "ntk", state_params)

# Given the a custom kernel and training/test data, compute the accuracy of an SVM 
def compute_svm_accuracy(custom_kernel_fn, analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test):
  svc = SVC(kernel=custom_kernel_fn)
  svc.fit(analysis_X_train, analysis_Y_train)
  predictions = svc.predict(analysis_X_test)
  accuracy = accuracy_score(analysis_Y_test, predictions)
  return accuracy

# Given the a custom kernel and training/test data, compute the accuracy of a Gaussian Process
def compute_gp_accuracy(custom_kernel_fn, analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test): 
  custom_gp_kernel = RBF(length_scale=1e7) # CustomKernel(kernel_fn=custom_kernel)
  gaussian_process_classifier = GaussianProcessRegressor(kernel=custom_gp_kernel)
  gaussian_process_classifier.fit(analysis_X_train, analysis_Y_train)
  predictions = gaussian_process_classifier.predict(analysis_X_test)
  accuracy = accuracy_score(analysis_Y_test, predictions)
  return accuracy 

# Compute the kernel alignment metric (Shan 2022: A Theory of Neural Tangent Kernel Alignment and Its Influence on Training)
@jax.jit
def compute_kernel_alignment(kernel, analysis_Y_test):
  kernel_alignment = (analysis_Y_test.T @ kernel @ analysis_Y_test) / (jnp.linalg.norm(kernel) * jnp.linalg.norm(analysis_Y_test))
  return kernel_alignment.item()

# Save the computed results to the determined file. Adding kernels is controlled by the add_kernel parameter as kernels take a large space to store
def save_results_to_json(steps, training_loss, test_loss, training_acc, test_acc, svm_accuracy, gp_accuracy, dots_results, 
                         kernel_alignments, computed_kernels, experiment_json_file_name, add_kernel): 
    
  graph_data = {
      "step": steps, 
      "training_loss": training_loss,
      "test_loss": test_loss,
      "training_acc": training_acc,
      "test_acc": test_acc,
      "svm_accuracy": svm_accuracy,
      "gp_accuracy": gp_accuracy,
      "dots": dots_results,
      "kernel_alignment": kernel_alignments,
  }

  if add_kernel:
    graph_data["kernels"] = computed_kernels 

  json_data = json.dumps(graph_data, indent=2)

  checkpoint_dir = Path(logs_base_path, "results", f"{experiment_json_file_name}.json")

  with open(checkpoint_dir, "w") as json_file:
    json_file.write(json_data)

"""
logs_base_path: e.g. "/home/dm894/idiots/logs/"
experiments: (experiment_name, experiment_json_file_name, experiment_checkpoint_path, experiment_type, step_distance, total_steps, num_dots_samples, num_analysis_training_samples, num_analysis_test_samples) list
  experiment_name: for printing out before experiment starts (does not affect the remainder of the program)
  experiment_json_file_name: e.g. "mnist-100" to save file as "mnist-100.json"
  experiment_checkpoint_path: e.g. "checkpoints/mnist-100/checkpoints"
  experiment_type: either "algorithmic" for modular division or S5, or "mnist" for mnist 
  step_distance: distance between checkpoints you want to analyse 
  total_steps = value of the highest checkpoint you want to analyse 
  num_kernel_samples = number of samples used when computing kernels (for DOTS and the remaining analysis) 
  num_analysis_training_samples = number of training samples used in the remaining analysis: when fitting the SVM, GP, and metrics such as kernel alignment 
  num_analysis_test_samples = number of test samples used in the remaining analysis: when fitting the SVM, GP, and metrics such as kernel alignment 
add_kernel: whether the kernel should be computed and added to the log file (can take up a large amount of space)
kernel_batch_size: batch size when computing the kernels using nt.batch
"""
def compute_results(logs_base_path, experiments, add_kernel, kernel_batch_size=32): 
  
  for experiment_name, experiment_json_file_name, experiment_checkpoint_path, experiment_type, step_distance, total_steps, num_kernel_samples, num_analysis_training_samples, num_analysis_test_samples in experiments: 
    
    print("Experiment:", experiment_name)

    experiment_checkpoint_path = Path(logs_base_path, experiment_checkpoint_path)

    if experiment_type == "algorithmic":
      restore_fn = algorithmic_restore
      restore_partial_fn = algorithmic_restore_partial
    elif experiment_type == "mnist":
      restore_fn = mnist_restore
      restore_partial_fn = mnist_restore_partial
    else: 
      print(f"Experiment type {experiment_type} not valid.")
      exit(1)

    restore_manager, _, _, ds_train, ds_test = restore_fn(experiment_checkpoint_path, 0)

    X_test, Y_test = jnp.array(ds_test['x']), jnp.array(ds_test['y'])

    num_y_classes = ds_train.features["y"].num_classes
    y_classes = jnp.array([i for i in range(num_y_classes)])

    df = extract_data_from_checkpoints(restore_manager, ds_train, ds_test, num_y_classes, total_steps, step_distance, restore_partial_fn)
    transformer_states, steps, training_loss, test_loss, training_acc, test_acc = parse_general_checkpoint_dataframe(df)

    svm_accuracy = []
    gp_accuracy = []
    dots_results = []
    computed_kernels = []
    kernel_alignments = []

    # kernel dataset is used for computing the kernels used in DOTS and the remaining analysis 
    # analysis datasets are used for the remaining analysis: SVM, GP, and remaining metrics such as kernel alignment 
    kernel_X, analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test = generate_kernel_and_analysis_datasets(X_test, Y_test, num_kernel_samples, num_analysis_training_samples, num_analysis_test_samples, experiment_type)

    for i, transformer_state in enumerate(transformer_states):
      gc.collect()
      print(f"Computing Results: {i + 1}/{len(transformer_states)}")

      kernel_trace_axes_fn = compute_kernel_trace_axes_fn(transformer_state.apply_fn)
      kernel_fn = compute_kernel_fn(transformer_state.apply_fn)

      kernel_trace_axes = compute_kernel_trace_axes(kernel_trace_axes_fn, transformer_state.params, kernel_X, kernel_batch_size)
      kernel = compute_kernel(kernel_fn, transformer_state.params, kernel_X, kernel_batch_size) 

      custom_kernel_fn = compute_custom_kernel_fn(kernel_fn, transformer_state.params)

      computed_kernels.append(kernel.tolist())
      dots_results.append(compute_dots(kernel_trace_axes))
      svm_accuracy.append(compute_svm_accuracy(custom_kernel_fn, analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test))
      gp_accuracy.append(compute_gp_accuracy(custom_kernel_fn, analysis_X_train, analysis_Y_train, analysis_X_test, analysis_Y_test))
      kernel_alignments.append(compute_kernel_alignment(kernel, analysis_Y_test))

    print("Storing Results...")
    save_results_to_json(steps, training_loss, test_loss, training_acc, test_acc, svm_accuracy, gp_accuracy, dots_results, 
                         kernel_alignments, computed_kernels, experiment_json_file_name, add_kernel)

if __name__ == "__main__":

  logs_base_path = "/home/dm894/idiots/logs/"

  experiments = [
                ("mnist", "mnist-slower", "checkpoints/mnist-slower/checkpoints", "mnist", 10_000, 100_000, 256, 256, 256),
                ]

  compute_results(logs_base_path, experiments, add_kernel=False)