from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.grokking.training import restore as grokking_restore, eval_step
from idiots.experiments.classification.training import restore as classification_restore
from idiots.utils import metrics
import neural_tangents as nt
from einops import rearrange
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import json
import os
import warnings
import gc

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

warnings.filterwarnings('ignore')

# --- Helper Functions ---

def eval_checkpoint(step, batch_size, checkpoint_dir, experiment_type):
  if experiment_type == "grokking":
    config, state, ds_train, ds_test = grokking_restore(checkpoint_dir, step)
  elif experiment_type == "classification":
    config, state, ds_train, ds_test = classification_restore(checkpoint_dir, step)
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

  train_loss, train_acc = eval_loss_acc(ds_train)
  test_loss, test_acc = eval_loss_acc(ds_test)

  return state, ds_train, ds_test, train_loss, train_acc, test_loss, test_acc


# --- Main Loop ---


logs_base_path = "../../../logs/"

# In form (experiment_name, experiment_checkpoint_path, experiment_type, svm_proportion_of_data, svm_training_data_proportion)

# experiment_type = "classification" or "grokking", they require different restore functions 

# The number of data samples the SVM is trained on = TEST_DATA_SIZE * svm_proportion_of_data * svm_training_data_proportion
# The number of data samples the SVM is tested on = TEST_DATA_SIZE * svm_proportion_of_data * (1 - svm_training_data_proportion)
# Note that the SVM is trained on the transformer test data

experiments = [("mnist", "mnist-tenth", "checkpoints/mnist/checkpoints", "classification", 100, 3000, 0.1, 0.5),
               ("mnist", "mnist-quarter", "checkpoints/mnist/checkpoints", "classification", 100, 3000, 0.25, 0.5),
               ("mnist", "mnist-half", "checkpoints/mnist/checkpoints", "classification", 100, 3000, 0.5, 0.5),
               ("mnist", "mnist-whole", "checkpoints/mnist/checkpoints", "classification", 100, 3000, 1, 0.5)
               ("div", "div", "checkpoints/division/checkpoints", "grokking", 1000, 50_000, 1, 0.5), 
               ("div_mse", "div_mse", "checkpoints/division_mse/checkpoints", "grokking", 1000, 50_000, 1, 0.5), 
               ("s5", "s5", "checkpoints/s5/checkpoints", "grokking", 1000, 50_000, 1, 0.5)]

for experiment_name, experiment_json_file_name, experiment_path, experiment_type, step_distance, total_epochs, svm_proportion_of_data, svm_training_data_proportion in experiments:

  print("Experiment:", experiment_name)

  checkpoint_dir = Path(logs_base_path, experiment_path)
  eval_checkpoint_batch_size = 512 # !!!!!

  # Extract data from checkpoints 
  data = []
  for step in range(0, total_epochs, step_distance): # !!!!!
    state, ds_train, ds_test, train_loss, train_acc, test_loss, test_acc = eval_checkpoint(step, eval_checkpoint_batch_size, checkpoint_dir, experiment_type)
    data.append(
        {
            "step": step,
            "state": state,
            "ds_train": ds_train,
            "ds_test": ds_test,
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

  training_loss = df_loss[df_loss["split"] == "train"]["loss"].tolist()
  test_loss = df_loss[df_loss["split"] == "test"]["loss"].tolist()
  training_acc = df_acc[df_acc["split"] == "train"]["accuracy"].tolist()
  test_acc = df_acc[df_acc["split"] == "test"]["accuracy"].tolist()

  df = pd.DataFrame(data)
  state_checkpoints = df['state'].tolist()
  train_data_checkpoints = df['ds_train'].tolist()
  test_data_checkpoints = df['ds_test'].tolist()

  # Compute SVM accuracy, dots and kernels from each recorded checkpoint  
  svm_accuracy = []
  dots_results = []
  computed_kernels = []

  X_test = jnp.array(test_data_checkpoints[0]['x'])
  Y_test = jnp.array(test_data_checkpoints[0]['y'])

  X_test_num_samples = len(X_test)
  batch_size = 32

  dots_num_samples = X_test_num_samples // batch_size * batch_size
  dots_X = X_test[:dots_num_samples]

  svm_train_num_samples = int(X_test_num_samples * svm_proportion_of_data * svm_training_data_proportion) // batch_size * batch_size 
  svm_X_train = X_test[:svm_train_num_samples]
  svm_Y_train = Y_test[:svm_train_num_samples]

  svm_test_num_samples = int(X_test_num_samples * svm_proportion_of_data * (1 - svm_training_data_proportion)) // batch_size * batch_size 
  svm_X_test = X_test[svm_train_num_samples:svm_train_num_samples+svm_test_num_samples]
  svm_Y_test = X_test[svm_train_num_samples:svm_train_num_samples+svm_test_num_samples]

  for i in range(len(state_checkpoints)):

    gc.collect()

    print(f"Iteration: {i}/{len(state_checkpoints)}")

    state = state_checkpoints[i]

    # Compute kernel and DOTS 

    kernel_fn = nt.empirical_kernel_fn(state.apply_fn,
                                       vmap_axes=0,
                                       trace_axes=(),
                                       implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)

    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size)
    kernel = kernel_fn_batched(dots_X, None, "ntk", state.params)

    computed_kernels.append(kernel.tolist())
    dots_results.append(jnp.linalg.matrix_rank(kernel).item())

    # Compute SVM accuracy 

    def custom_kernel(X1, X2):
      kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size)
      return kernel_fn_batched(X1, X2, "ntk", state.params)

    svc = SVC(kernel=custom_kernel)

    svc.fit(svm_X_train, svm_Y_train)

    predictions = svc.predict(svm_X_test)
    accuracy = accuracy_score(svm_Y_test, predictions)

    svm_accuracy.append(accuracy)

  # Store results as a JSON file 
  graph_data = {
      "training_loss": training_loss,
      "test_loss": test_loss,
      "training_acc": training_acc,
      "test_acc": test_acc,
      "svm_accuracy": svm_accuracy,
      "dots": dots_results,
      "kernel": computed_kernels, 
  }

  json_data = json.dumps(graph_data, indent=2)

  checkpoint_dir = Path(logs_base_path, "results", f"{experiment_json_file_name}.json")

  with open(checkpoint_dir, "w") as json_file:
    json_file.write(json_data)
