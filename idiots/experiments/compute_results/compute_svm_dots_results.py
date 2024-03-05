from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.grokking.training import restore as grokking_restore, restore_partial as grokking_restore_partial, eval_step
from idiots.experiments.classification.training import restore as classification_restore, restore_partial as classification_restore_partial
from idiots.utils import metrics
import neural_tangents as nt
from einops import rearrange
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score
import json
import os
import warnings
import gc
from sklearn.model_selection import train_test_split

TEST_MODE = False 
ADD_KERNEL = False 

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

warnings.filterwarnings('ignore')

# --- Helper Functions ---


def eval_checkpoint(step, batch_size, experiment_type, ds_train, ds_test, num_classes, mngr):
  if experiment_type == "grokking":
    config, state = grokking_restore_partial(mngr, step, ds_train["x"][:1], num_classes)
  elif experiment_type == "classification":
    config, state = classification_restore_partial(mngr, step, ds_train["x"][:1], num_classes)
  else:
    print(f"Experiment type {experiment_type} not valid.")
    exit(1)

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

### --- Define GP kernel --- 

class CustomKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn
        super().__init__()

    def __call__(self, X, Y=None, eval_gradient=False):
        return self.kernel_fn(X, Y)

# --- Main Loop ---


logs_base_path = "/home/dm894/idiots/logs/"

# In form (experiment_name, experiment_checkpoint_path, experiment_type, step_distance, total_steps, num_dots_samples, num_svm_training_samples, num_svm_test_samples)

# step_distance = distance between checkpoints
# total_steps = value of the highest checkpoint

experiments = [
              ("division", "division", "checkpoints/division/checkpoints", "grokking", 1000, 10_000, 256, 64, 256),
              ]

for experiment_name, experiment_json_file_name, experiment_path, experiment_type, step_distance, total_steps, num_dots_samples, num_svm_training_samples, num_svm_test_samples in experiments:
 
  step_distance = step_distance if not TEST_MODE else total_steps # Only take one total step in TEST_MODE
  eval_checkpoint_batch_size = 512 if not TEST_MODE else 5 
  num_dots_samples = num_dots_samples if not TEST_MODE else 5
  num_svm_training_samples = num_svm_training_samples if not TEST_MODE else 5
  num_svm_test_samples = num_svm_test_samples if not TEST_MODE else 5

  print("Experiment:", experiment_name)

  checkpoint_dir = Path(logs_base_path, experiment_path)

  if experiment_type == "grokking":
    mngr, _, _, ds_train, ds_test = grokking_restore(checkpoint_dir, 0)
  elif experiment_type == "classification":
    mngr, _, _, ds_train, ds_test = classification_restore(checkpoint_dir, 0)
  else:
    print(f"Experiment type {experiment_type} not valid.")
    exit(1)

  X_test = jnp.array(ds_test['x'])
  Y_test = jnp.array(ds_test['y'])

  num_classes = ds_train.features["y"].num_classes

  # Extract data from checkpoints
  data = []
  for step in range(0, total_steps, step_distance):

    print(f"Loading Data: {(step // step_distance) + 1}/{total_steps // step_distance}")

    state, train_loss, train_acc, test_loss, test_acc = eval_checkpoint(step, eval_checkpoint_batch_size, experiment_type, ds_train, ds_test, num_classes, mngr)
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

  # Get state and train/test data lists
  df = pd.DataFrame(data)

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

  df = pd.DataFrame(data)
  state_checkpoints = df['state'].tolist()

  # Compute SVM accuracy, dots and kernels from each recorded checkpoint
  svm_accuracy = []
  gp_accuracy = []
  dots_results = []
  computed_kernels = []
  kernel_alignments = []

  batch_size = 32

  dots_X = X_test[:num_dots_samples]

  # svm_X_train, svm_X_test, svm_Y_train, svm_Y_test = train_test_split(X_test, Y_test, test_size=num_svm_test_samples, stratify=Y_test)

  svm_X_train = X_test[:num_svm_training_samples]
  svm_Y_train = Y_test[:num_svm_training_samples]

  svm_X_test = X_test[num_svm_training_samples:num_svm_training_samples+num_svm_test_samples]
  svm_Y_test = Y_test[num_svm_training_samples:num_svm_training_samples+num_svm_test_samples]

  kernel_fn_trace = nt.empirical_kernel_fn(state.apply_fn,
                       vmap_axes=0,
                       trace_axes=(),
                       implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)

  kernel_fn = nt.empirical_kernel_fn(state.apply_fn,
                     vmap_axes=0,
                     implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)
  
  @jax.jit 
  def calculate_kernel_rank(kernel_trace, eps=1e-5):
    return jnp.linalg.matrix_rank(kernel_trace, tol=eps)

  for i in range(len(state_checkpoints)):

    gc.collect()

    print(f"Computing Results: {i + 1}/{len(state_checkpoints)}")

    state = state_checkpoints[i]

    # Compute kernel and DOTS

    kernel_fn_trace_batched = nt.batch(kernel_fn_trace, device_count=-1, batch_size=batch_size)
    kernel_trace = kernel_fn_trace_batched(dots_X, None, "ntk", state.params)
    kernel_trace = rearrange(kernel_trace, "b1 b2 d1 d2 -> (b1 d1) (b2 d2)")
    kernel_rank = calculate_kernel_rank(kernel_trace)
    print(kernel_rank.item())
    dots_results.append(kernel_rank.item())

    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size)
    kernel = kernel_fn_batched(dots_X, None, "ntk", state.params)
    if ADD_KERNEL: 
      computed_kernels.append(kernel.tolist())

    # Compute SVM accuracy

    def custom_kernel(X1, X2):
      kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size)
      return kernel_fn_batched(X1, X2, "ntk", state.params)

    svc = SVC(kernel=custom_kernel)

    svc.fit(svm_X_train, svm_Y_train)

    predictions = svc.predict(svm_X_test)

    accuracy = accuracy_score(svm_Y_test, predictions)

    svm_accuracy.append(accuracy)

    # Compute GP accuracy 

    custom_gp_kernel = CustomKernel(kernel_fn=custom_kernel)

    if ADD_KERNEL:
      svm_X_train = rearrange(svm_X_train, 'b h w -> b (h w)')
      svm_X_test = rearrange(svm_X_test, 'b h w -> b (h w)')

    gaussian_process_classifier = GaussianProcessClassifier(kernel=custom_gp_kernel)
    gaussian_process_classifier.fit(svm_X_train, svm_Y_train)

    predictions = gaussian_process_classifier.predict(svm_X_test)

    accuracy = accuracy_score(svm_Y_test, predictions)

    gp_accuracy.append(accuracy)

    # Compute kernel alignment 

    kernel_alignment = (svm_Y_test.T @ kernel @ svm_Y_test) / (jnp.linalg.norm(kernel) * jnp.linalg.norm(svm_Y_test))

    kernel_alignments.append(kernel_alignment.item())

  print("Storing Results...")

  # Store results as a JSON file
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

  if ADD_KERNEL:
    graph_data["kernels"] = computed_kernels 

  json_data = json.dumps(graph_data, indent=2)

  checkpoint_dir = Path(logs_base_path, "results", f"{experiment_json_file_name}.json")

  with open(checkpoint_dir, "w") as json_file:
    json_file.write(json_data)
