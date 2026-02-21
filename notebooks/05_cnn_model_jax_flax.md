# CNN com JAX/Flax para CIFAR-10

## 1. Imports e paths


```python
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from flax import serialization

# Ensure project root is on sys.path so `utils` can be imported
cwd = Path.cwd()
if cwd.name.lower() == "notebooks":
    project_root = cwd.parent
elif cwd.name.lower() == "cifar10_project":
    project_root = cwd
elif (cwd / "cifar10_project").is_dir():
    project_root = cwd / "cifar10_project"
else:
    project_root = None

if project_root is not None and str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.ml_utils import (
    ensure_label_indices,
    compute_classification_metrics,
    compute_confusion_matrix,
    build_classification_report,
    log_metrics_to_csv,
    format_duration,
    plot_confusion_matrix_notebook01,
    plot_training_curves_notebook01,
)

print('JAX devices:', jax.devices())
print('JAX default backend:', jax.default_backend())
```

    JAX devices: [CudaDevice(id=0)]
    JAX default backend: gpu



```python
PROJECT_ROOT = os.getcwd()
if os.path.basename(PROJECT_ROOT).lower() == 'notebooks':
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
elif os.path.basename(PROJECT_ROOT).lower() != 'cifar10_project':
    candidate = os.path.join(PROJECT_ROOT, 'cifar10_project')
    if os.path.isdir(candidate):
        PROJECT_ROOT = candidate

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

data_path = os.path.join(PROCESSED_DIR, 'cifar10_processed.npz')
performance_report_path = os.path.join(DATA_DIR, 'model_performance_report.csv')
print('Data path:', data_path)
print('Performance report path:', performance_report_path)
```

    Data path: /mnt/c/Users/User/Documents/Bootcamp AI and Data Science/Linux/DataScience_ironhack/Week6/project/cifar10_project/data/processed/cifar10_processed.npz
    Performance report path: /mnt/c/Users/User/Documents/Bootcamp AI and Data Science/Linux/DataScience_ironhack/Week6/project/cifar10_project/data/model_performance_report.csv


## 2. Carregar dados


```python
data = np.load(data_path)

x_train = data['x_train'].astype(np.float32)
y_train = ensure_label_indices(data['y_train']).astype(np.int32)
x_val = data['x_val'].astype(np.float32)
y_val = ensure_label_indices(data['y_val']).astype(np.int32)
x_test = data['x_test'].astype(np.float32)
y_test = ensure_label_indices(data['y_test']).astype(np.int32)

print('Train:', x_train.shape, y_train.shape)
print('Val:  ', x_val.shape, y_val.shape)
print('Test: ', x_test.shape, y_test.shape)

# Garantir formato [0,1] como no pipeline atual
if x_train.max() > 1.0:
    x_train = x_train / 255.0
    x_val = x_val / 255.0
    x_test = x_test / 255.0
```

    Train: (40000, 32, 32, 3) (40000,)
    Val:   (10000, 32, 32, 3) (10000,)
    Test:  (10000, 32, 32, 3) (10000,)


## 3. Augmentation e batching (padrão do notebook 04)


```python
BATCH_SIZE = 64
NUM_CLASSES = 10
SEED = 42

def augment_batch(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = images.copy()

    # random_flip_left_right
    flip_mask = rng.random(out.shape[0]) < 0.5
    out[flip_mask] = out[flip_mask, :, ::-1, :]

    # random_brightness(max_delta=0.1)
    brightness = rng.uniform(-0.1, 0.1, size=(out.shape[0], 1, 1, 1)).astype(np.float32)
    out = out + brightness

    # random_contrast(lower=0.9, upper=1.1)
    contrast = rng.uniform(0.9, 1.1, size=(out.shape[0], 1, 1, 1)).astype(np.float32)
    mean = out.mean(axis=(1, 2), keepdims=True)
    out = (out - mean) * contrast + mean

    return np.clip(out, 0.0, 1.0)

def iterate_minibatches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    augment: bool = False,
    rng: np.random.Generator | None = None,
):
    n = x.shape[0]
    indices = np.arange(n)

    if shuffle:
        (rng or np.random.default_rng()).shuffle(indices)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        xb = x[batch_idx]
        yb = y[batch_idx]

        if augment:
            xb = augment_batch(xb, rng or np.random.default_rng())

        yield xb.astype(np.float32), yb.astype(np.int32)
```

## 4. Modelo Flax: Cifar10CNN


```python
class Cifar10CNN(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Input: 32x32x3

        # Bloco 1: Conv(32) + ReLU, MaxPool
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Bloco 2: [Conv(64) + BN + ReLU] x2, MaxPool
        for _ in range(2):
            x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
            x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Bloco 3: [Conv(128) + BN + ReLU] x2, MaxPool
        for _ in range(2):
            x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
            x = nn.gelu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding='SAME')

        # Bloco 4: [Conv(256) + BN + ReLU] x2
        for _ in range(2):
            x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not train, momentum=0.9, epsilon=1e-5)(x)
            x = nn.gelu(x)

        # Global Average Pooling
        x = jnp.mean(x, axis=(1, 2))

        # Dense(10)
        x = nn.Dense(self.num_classes)(x)
        return x
```

## 5. TrainState, create_train_state, train_step, eval_step


```python
class TrainState(train_state.TrainState):
    batch_stats: Dict

def create_train_state(rng, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
    model = Cifar10CNN(num_classes=NUM_CLASSES)
    variables = model.init(rng, jnp.ones((1, 32, 32, 3), dtype=jnp.float32), train=True)
    params = variables['params']
    batch_stats = variables['batch_stats']

    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )

@jax.jit
def train_step(state: TrainState, images: jnp.ndarray, labels: jnp.ndarray):
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images,
            train=True,
            mutable=['batch_stats'],
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        return loss, (logits, updates)

    (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])

    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    metrics = {'loss': loss, 'accuracy': acc}
    return state, metrics

@jax.jit
def eval_step(state: TrainState, images: jnp.ndarray, labels: jnp.ndarray):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images,
        train=False,
        mutable=False,
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    metrics = {'loss': loss, 'accuracy': acc}
    return metrics, logits
```

## 6. Loop de treino simples (com early stopping)


```python
def run_epoch(state, x, y, batch_size, train: bool, rng: np.random.Generator):
    losses = []
    accs = []

    for xb, yb in iterate_minibatches(x, y, batch_size=batch_size, shuffle=train, augment=train, rng=rng):
        xb_j = jnp.asarray(xb)
        yb_j = jnp.asarray(yb)

        if train:
            state, metrics = train_step(state, xb_j, yb_j)
        else:
            metrics, _ = eval_step(state, xb_j, yb_j)

        losses.append(float(metrics['loss']))
        accs.append(float(metrics['accuracy']))

    return state, float(np.mean(losses)), float(np.mean(accs))

rng = np.random.default_rng(SEED)
jax_rng = jax.random.PRNGKey(SEED)

learning_rate = 1e-3
#weight_decay = 0
epochs = 200
patience = 32

state = create_train_state(jax_rng, learning_rate=learning_rate)#, weight_decay=weight_decay)

history = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'learning_rate': [],
}

best_val_acc = 0.0
best_epoch = 0
best_snapshot = serialization.to_bytes({'params': state.params, 'batch_stats': state.batch_stats})

start_time = time.time()

for epoch in range(1, epochs + 1):
    state, train_loss, train_acc = run_epoch(state, x_train, y_train, BATCH_SIZE, train=True, rng=rng)
    state, val_loss, val_acc = run_epoch(state, x_val, y_val, BATCH_SIZE, train=False, rng=rng)

    history['loss'].append(train_loss)
    history['accuracy'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)
    history['learning_rate'].append(learning_rate)

    improved = val_acc > best_val_acc
    if improved:
        best_val_acc = val_acc
        best_epoch = epoch
        best_snapshot = serialization.to_bytes({'params': state.params, 'batch_stats': state.batch_stats})

    print(
        f"Epoch {epoch:03d}/{epochs} | "
        f"loss={train_loss:.4f} acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )

    if (epoch - best_epoch) >= patience:
        print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch}).")
        break

train_time_sec = time.time() - start_time
print(f"Training time: {format_duration(train_time_sec)} ({train_time_sec:.2f} sec)")

best_variables = serialization.from_bytes(
    {'params': state.params, 'batch_stats': state.batch_stats},
    best_snapshot,
)
state = state.replace(params=best_variables['params'], batch_stats=best_variables['batch_stats'])
print(f"Restored best validation checkpoint from epoch {best_epoch}.")
```

    Epoch 001/200 | loss=1.2373 acc=0.5567 | val_loss=1.0648 val_acc=0.6171
    Epoch 002/200 | loss=0.8180 acc=0.7143 | val_loss=0.8659 val_acc=0.6993
    Epoch 003/200 | loss=0.6595 acc=0.7702 | val_loss=0.7201 val_acc=0.7590
    Epoch 004/200 | loss=0.5578 acc=0.8075 | val_loss=0.6442 val_acc=0.7794
    Epoch 005/200 | loss=0.4834 acc=0.8323 | val_loss=0.6083 val_acc=0.7905
    Epoch 006/200 | loss=0.4210 acc=0.8558 | val_loss=0.5415 val_acc=0.8139
    Epoch 007/200 | loss=0.3693 acc=0.8711 | val_loss=0.5540 val_acc=0.8129
    Epoch 008/200 | loss=0.3279 acc=0.8867 | val_loss=0.5670 val_acc=0.8173
    Epoch 009/200 | loss=0.2780 acc=0.9029 | val_loss=0.5495 val_acc=0.8226
    Epoch 010/200 | loss=0.2519 acc=0.9125 | val_loss=0.5310 val_acc=0.8373
    Epoch 011/200 | loss=0.2140 acc=0.9249 | val_loss=0.5620 val_acc=0.8308
    Epoch 012/200 | loss=0.1921 acc=0.9332 | val_loss=0.5608 val_acc=0.8338
    Epoch 013/200 | loss=0.1662 acc=0.9414 | val_loss=0.6394 val_acc=0.8177
    Epoch 014/200 | loss=0.1490 acc=0.9487 | val_loss=0.5987 val_acc=0.8341
    Epoch 015/200 | loss=0.1255 acc=0.9565 | val_loss=0.6393 val_acc=0.8328
    Epoch 016/200 | loss=0.1237 acc=0.9571 | val_loss=0.6757 val_acc=0.8272
    Epoch 017/200 | loss=0.1032 acc=0.9638 | val_loss=0.6419 val_acc=0.8339
    Epoch 018/200 | loss=0.0963 acc=0.9666 | val_loss=0.6411 val_acc=0.8383
    Epoch 019/200 | loss=0.0918 acc=0.9681 | val_loss=0.6850 val_acc=0.8320
    Epoch 020/200 | loss=0.0832 acc=0.9709 | val_loss=0.7693 val_acc=0.8158
    Epoch 021/200 | loss=0.0779 acc=0.9727 | val_loss=0.6961 val_acc=0.8367
    Epoch 022/200 | loss=0.0755 acc=0.9741 | val_loss=0.7015 val_acc=0.8344
    Epoch 023/200 | loss=0.0691 acc=0.9755 | val_loss=0.7519 val_acc=0.8256
    Epoch 024/200 | loss=0.0676 acc=0.9766 | val_loss=0.7649 val_acc=0.8213
    Epoch 025/200 | loss=0.0615 acc=0.9787 | val_loss=0.7133 val_acc=0.8398
    Epoch 026/200 | loss=0.0544 acc=0.9811 | val_loss=0.7521 val_acc=0.8400
    Epoch 027/200 | loss=0.0563 acc=0.9807 | val_loss=0.8473 val_acc=0.8267
    Epoch 028/200 | loss=0.0588 acc=0.9797 | val_loss=0.7429 val_acc=0.8417
    Epoch 029/200 | loss=0.0480 acc=0.9826 | val_loss=0.8937 val_acc=0.8214
    Epoch 030/200 | loss=0.0501 acc=0.9822 | val_loss=0.8033 val_acc=0.8374
    Epoch 031/200 | loss=0.0459 acc=0.9837 | val_loss=0.7696 val_acc=0.8383
    Epoch 032/200 | loss=0.0467 acc=0.9836 | val_loss=0.7904 val_acc=0.8330
    Epoch 033/200 | loss=0.0451 acc=0.9836 | val_loss=0.8604 val_acc=0.8256
    Epoch 034/200 | loss=0.0411 acc=0.9852 | val_loss=0.8124 val_acc=0.8367
    Epoch 035/200 | loss=0.0354 acc=0.9879 | val_loss=0.7959 val_acc=0.8399
    Epoch 036/200 | loss=0.0379 acc=0.9867 | val_loss=0.8446 val_acc=0.8361
    Epoch 037/200 | loss=0.0372 acc=0.9869 | val_loss=0.8581 val_acc=0.8414
    Epoch 038/200 | loss=0.0399 acc=0.9859 | val_loss=0.8641 val_acc=0.8318
    Epoch 039/200 | loss=0.0408 acc=0.9854 | val_loss=0.8148 val_acc=0.8411
    Epoch 040/200 | loss=0.0314 acc=0.9894 | val_loss=0.8305 val_acc=0.8426
    Epoch 041/200 | loss=0.0362 acc=0.9882 | val_loss=0.8209 val_acc=0.8442
    Epoch 042/200 | loss=0.0310 acc=0.9898 | val_loss=0.8451 val_acc=0.8420
    Epoch 043/200 | loss=0.0333 acc=0.9885 | val_loss=0.8750 val_acc=0.8372
    Epoch 044/200 | loss=0.0340 acc=0.9880 | val_loss=0.8746 val_acc=0.8396
    Epoch 045/200 | loss=0.0377 acc=0.9875 | val_loss=0.8603 val_acc=0.8426
    Epoch 046/200 | loss=0.0257 acc=0.9911 | val_loss=0.8847 val_acc=0.8363
    Epoch 047/200 | loss=0.0267 acc=0.9908 | val_loss=0.9335 val_acc=0.8325
    Epoch 048/200 | loss=0.0322 acc=0.9890 | val_loss=0.8943 val_acc=0.8367
    Epoch 049/200 | loss=0.0303 acc=0.9893 | val_loss=0.8968 val_acc=0.8427
    Epoch 050/200 | loss=0.0281 acc=0.9906 | val_loss=0.8908 val_acc=0.8391
    Epoch 051/200 | loss=0.0255 acc=0.9911 | val_loss=0.8669 val_acc=0.8397
    Epoch 052/200 | loss=0.0223 acc=0.9927 | val_loss=0.9480 val_acc=0.8397
    Epoch 053/200 | loss=0.0281 acc=0.9900 | val_loss=0.8699 val_acc=0.8430
    Epoch 054/200 | loss=0.0265 acc=0.9912 | val_loss=0.8718 val_acc=0.8386
    Epoch 055/200 | loss=0.0256 acc=0.9908 | val_loss=0.8255 val_acc=0.8524
    Epoch 056/200 | loss=0.0210 acc=0.9928 | val_loss=0.9182 val_acc=0.8379
    Epoch 057/200 | loss=0.0209 acc=0.9926 | val_loss=0.9000 val_acc=0.8459
    Epoch 058/200 | loss=0.0234 acc=0.9919 | val_loss=0.8954 val_acc=0.8439
    Epoch 059/200 | loss=0.0266 acc=0.9907 | val_loss=0.9035 val_acc=0.8409
    Epoch 060/200 | loss=0.0237 acc=0.9914 | val_loss=0.8580 val_acc=0.8473
    Epoch 061/200 | loss=0.0224 acc=0.9926 | val_loss=0.8844 val_acc=0.8475
    Epoch 062/200 | loss=0.0222 acc=0.9928 | val_loss=0.9360 val_acc=0.8412
    Epoch 063/200 | loss=0.0207 acc=0.9927 | val_loss=0.8588 val_acc=0.8468
    Epoch 064/200 | loss=0.0164 acc=0.9946 | val_loss=0.9423 val_acc=0.8430
    Epoch 065/200 | loss=0.0235 acc=0.9915 | val_loss=0.9205 val_acc=0.8444
    Epoch 066/200 | loss=0.0210 acc=0.9925 | val_loss=0.9333 val_acc=0.8403
    Epoch 067/200 | loss=0.0225 acc=0.9922 | val_loss=0.9173 val_acc=0.8452
    Epoch 068/200 | loss=0.0207 acc=0.9929 | val_loss=0.8917 val_acc=0.8444
    Epoch 069/200 | loss=0.0209 acc=0.9927 | val_loss=0.8883 val_acc=0.8448
    Epoch 070/200 | loss=0.0171 acc=0.9940 | val_loss=0.9491 val_acc=0.8411
    Epoch 071/200 | loss=0.0178 acc=0.9940 | val_loss=0.9084 val_acc=0.8446
    Epoch 072/200 | loss=0.0182 acc=0.9940 | val_loss=0.9425 val_acc=0.8420
    Epoch 073/200 | loss=0.0199 acc=0.9931 | val_loss=1.0449 val_acc=0.8300
    Epoch 074/200 | loss=0.0138 acc=0.9950 | val_loss=0.8823 val_acc=0.8515
    Epoch 075/200 | loss=0.0183 acc=0.9933 | val_loss=0.9848 val_acc=0.8358
    Epoch 076/200 | loss=0.0201 acc=0.9936 | val_loss=0.9347 val_acc=0.8477
    Epoch 077/200 | loss=0.0188 acc=0.9935 | val_loss=0.9680 val_acc=0.8410
    Epoch 078/200 | loss=0.0152 acc=0.9950 | val_loss=0.9436 val_acc=0.8441
    Epoch 079/200 | loss=0.0140 acc=0.9950 | val_loss=0.9776 val_acc=0.8457
    Epoch 080/200 | loss=0.0165 acc=0.9942 | val_loss=1.0068 val_acc=0.8381
    Epoch 081/200 | loss=0.0193 acc=0.9929 | val_loss=1.0067 val_acc=0.8381
    Epoch 082/200 | loss=0.0153 acc=0.9946 | val_loss=0.9623 val_acc=0.8410


## 7. Avaliação no teste, CSV e modelo


```python
def predict_dataset(state: TrainState, x: np.ndarray, y: np.ndarray, batch_size: int):
    all_preds = []
    losses = []
    accs = []

    for xb, yb in iterate_minibatches(x, y, batch_size=batch_size, shuffle=False, augment=False):
        metrics, logits = eval_step(state, jnp.asarray(xb), jnp.asarray(yb))
        losses.append(float(metrics['loss']))
        accs.append(float(metrics['accuracy']))
        all_preds.append(np.asarray(jnp.argmax(logits, axis=-1)))

    return np.concatenate(all_preds), float(np.mean(losses)), float(np.mean(accs))

y_pred, test_loss, test_acc = predict_dataset(state, x_test, y_test, BATCH_SIZE)
print(f"JAX/Flax test accuracy: {test_acc:.4f}")
print(f"JAX/Flax test loss: {test_loss:.4f}")

cls_metrics = compute_classification_metrics(y_test, y_pred, average='macro')

row = {
    'model_name': 'CNN - JAX/Flax',
    'num_params': int(sum(np.prod(np.array(p.shape)) for p in jax.tree_util.tree_leaves(state.params))),
    'train_time_sec': round(float(train_time_sec), 2),
    'train_time': format_duration(train_time_sec),
    'timestamp': time.strftime('%Y-%m-%d %H:%M'),
    'accuracy': float(cls_metrics['accuracy']),
    'precision': float(cls_metrics['precision']),
    'recall': float(cls_metrics['recall']),
    'f1': float(cls_metrics['f1']),
    'support': int(cls_metrics['support']),
    'final_accuracy': float(history['accuracy'][-1]),
    'final_loss': float(history['loss'][-1]),
    'final_val_accuracy': float(history['val_accuracy'][-1]),
    'final_val_loss': float(history['val_loss'][-1]),
    'final_learning_rate': float(history['learning_rate'][-1]),
    'epochs_trained': int(len(history['accuracy'])),
    'test_loss': float(test_loss),
}

log_metrics_to_csv(performance_report_path, row, append=True)
print('Logged metrics:', row)

model_path = os.path.join(MODELS_DIR, 'cifar10_cnn_jax_flax.msgpack')
with open(model_path, 'wb') as f:
    f.write(serialization.to_bytes({'params': state.params, 'batch_stats': state.batch_stats}))
print('Saved JAX/Flax model to:', model_path)
```

## 8. Relatório e gráficos (padrão do projeto)


```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print('Classification Report (JAX/Flax):')
print(build_classification_report(y_test, y_pred, target_names=class_names))

cm = compute_confusion_matrix(y_test, y_pred)
plot_confusion_matrix_notebook01(
    confusion=cm,
    class_names=class_names,
    title='Confusion Matrix - CNN JAX/Flax',
    save_path=os.path.join(REPORTS_DIR, 'jax_flax_confusion.png'),
)

plot_training_curves_notebook01(
    history=history,
    save_path=os.path.join(REPORTS_DIR, 'jax_flax_curves.png'),
)

print('Saved plots to reports/: jax_flax_confusion.png, jax_flax_curves.png')
```
