from pathlib import Path

import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess

sys.path.append("/home/dc755/idiots")


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

def eval_checkpoint(step):
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

def get_dots(kernel_fn, X):
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=32)
    kernel = kernel_fn_batched(X, None, "ntk", state.params)
    return jnp.linalg.matrix_rank(kernel).item()

experiments = [("div", "../idiots/logs/division/exp21/checkpoints"), ("div_mse", "../idiots/logs/division_mse/exp22/checkpoints"), ("s5", "../idiots/logs/s5/exp24/checkpoints")]

for experiment_name, experiment_path in experiments[1:]: 

  checkpoint_dir = Path(experiment_path)
  batch_size = 512

  data = []
  for step in range(0, 50000, 1000):
      state, ds_train, ds_test, train_loss, train_acc, test_loss, test_acc = eval_checkpoint(step)
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

  svm_accuracy = []
  dots_results = []

  N_train = 512
  N_test = 512

  X_train = jnp.array(train_data_checkpoints[0]['x'][:N_train])
  Y_train = jnp.array(train_data_checkpoints[0]['y'][:N_train])

  X_test = jnp.array(test_data_checkpoints[0]['x'][:N_test])
  Y_test = jnp.array(test_data_checkpoints[0]['y'][:N_test])

  for i in range(len(state_checkpoints)):

    gc.collect() 

    print(f"Iteration: {i}/{len(state_checkpoints)}")

    state = state_checkpoints[i]

    kernel_fn = nt.empirical_kernel_fn(state.apply_fn, 
        vmap_axes=0,
        implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES,)
    
    dots = get_dots(kernel_fn, X_test)
    
    def custom_kernel(X1, X2):
      kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=32)
      return kernel_fn_batched(X1, X2, "ntk", state.params)

    svc = SVC(kernel=custom_kernel)
    
    svc.fit(X_train, Y_train)

    predictions = svc.predict(X_test)
    accuracy = accuracy_score(Y_test, predictions)

    svm_accuracy.append(accuracy)
    dots_results.append(dots)

  graph_data = {
    "training_loss": training_loss,
    "test_loss": test_loss,
    "training_acc": train_acc, 
    "test_acc": test_acc,
    "svm_accuracy": svm_accuracy, 
    "dots": dots_results
  }

  json_data = json.dumps(graph_data, indent=2)

  with open(f"results_{experiment_name}.json", "w") as json_file:
      json_file.write(json_data)