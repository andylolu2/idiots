from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from idiots.dataset.dataloader import DataLoader
from idiots.experiments.grokking.training import restore, eval_step
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

def eval_checkpoint(step, batch_size):
  config, state, ds_train, ds_test = restore(checkpoint_dir, step)

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

experiments = [("mnist", "checkpoints/mnist/checkpoints")] # [("div", "checkpoints/division/checkpoints"), ("div_mse", "checkpoints/division_mse/checkpoints"), ("s5", "checkpoints/s5/checkpoints"), ("mnist", "checkpoints/mnist/checkpoints")]

for experiment_name, experiment_path in experiments:

  checkpoint_dir = Path(logs_base_path, experiment_path)
  eval_checkpoint_batch_size = 5 # 512 !!!!!

  # Extract data from checkpoints 
  data = []
  for step in range(0, 50000, 40000): # 1000 !!!!!
    state, ds_train, ds_test, train_loss, train_acc, test_loss, test_acc = eval_checkpoint(step, eval_checkpoint_batch_size)
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

  N_train = 5 #512 !!!!!
  N_test = 5 #512 !!!!! 

  X_test_full = jnp.array(test_data_checkpoints[0]['x'][:N_test])
  Y_test_full = jnp.array(test_data_checkpoints[0]['y'][:N_test])

  X_test_for_svm_training = X_test_full[:len(X_test_full) // 2]
  Y_test_for_svm_training = Y_test_full[:len(Y_test_full) // 2]

  X_test_for_svm_testing = X_test_full[len(X_test_full) // 2:]
  Y_test_for_svm_testing = Y_test_full[len(Y_test_full) // 2:]

  # X_train = jnp.array(train_data_checkpoints[0]['x'][:N_train])
  # Y_train = jnp.array(train_data_checkpoints[0]['y'][:N_train])

  # X_test = jnp.array(test_data_checkpoints[0]['x'][:N_test])
  # Y_test = jnp.array(test_data_checkpoints[0]['y'][:N_test])

  for i in range(len(state_checkpoints)):

    gc.collect()

    print(f"Iteration: {i}/{len(state_checkpoints)}")

    state = state_checkpoints[i]

    # Compute kernel and DOTS 

    kernel_fn = nt.empirical_kernel_fn(state.apply_fn,
                                       vmap_axes=0,
                                       implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)

    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=32)
    kernel = kernel_fn_batched(X_test_full, None, "ntk", state.params)

    computed_kernels.append(kernel.tolist())
    dots_results.append(jnp.linalg.matrix_rank(kernel).item())

    # Compute SVM accuracy 

    def custom_kernel(X1, X2):
      kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=32)
      return kernel_fn_batched(X1, X2, "ntk", state.params)

    svc = SVC(kernel=custom_kernel)

    svc.fit(X_test_for_svm_training, Y_test_for_svm_training)

    predictions = svc.predict(X_test_for_svm_testing)
    accuracy = accuracy_score(Y_test_for_svm_testing, predictions)

    svm_accuracy.append(accuracy)

  # Store results as a JSON file 
  graph_data = {
      "training_loss": training_loss,
      "test_loss": test_loss,
      "training_acc": train_acc,
      "test_acc": test_acc,
      "svm_accuracy": svm_accuracy,
      "dots": dots_results,
      "kernel": computed_kernels, 
  }

  json_data = json.dumps(graph_data, indent=2)

  checkpoint_dir = Path(logs_base_path, "results", f"{experiment_name}.json")

  with open(checkpoint_dir, "w") as json_file:
    json_file.write(json_data)
