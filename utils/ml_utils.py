from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

ArrayLike = Union[np.ndarray, Sequence[float]]

PERFORMANCE_REPORT_COLUMNS = [
    "model_name",
    "num_params",
    "train_time_sec",
    "train_time",
    "timestamp",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "support",
    "final_accuracy",
    "final_loss",
    "final_val_accuracy",
    "final_val_loss",
    "final_learning_rate",
    "epochs_trained",
    "test_loss",
]


def format_duration(seconds: Optional[float]) -> Optional[str]:
    """Format seconds as a compact human-readable string, e.g. '7 min 02 sec'."""
    if seconds is None:
        return None

    total_seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours} h {minutes:02d} min {secs:02d} sec"
    return f"{minutes} min {secs:02d} sec"


def normalize_images(
    images: ArrayLike,
    scale: float = 255.0,
    dtype: str = "float32",
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> np.ndarray:
    """Normalize images to a numeric range, e.g. [0, 1]."""
    if scale <= 0:
        raise ValueError("scale must be > 0")

    out = np.asarray(images).astype(dtype) / scale
    if clip_range is not None:
        out = np.clip(out, clip_range[0], clip_range[1])
    return out


def normalize_splits(
    splits: Mapping[str, ArrayLike],
    scale: float = 255.0,
    dtype: str = "float32",
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Dict[str, np.ndarray]:
    """Normalize multiple dataset splits at once (train/val/test or any custom names)."""
    return {
        split_name: normalize_images(values, scale=scale, dtype=dtype, clip_range=clip_range)
        for split_name, values in splits.items()
    }


def ensure_label_indices(labels: ArrayLike) -> np.ndarray:
    """Convert labels to class indices if they are one-hot encoded."""
    y = np.asarray(labels)
    if y.ndim > 1 and y.shape[-1] > 1:
        return np.argmax(y, axis=-1)
    return y.reshape(-1)


def random_cutout(
    image: np.ndarray,
    max_holes: int = 1,
    min_size: int = 4,
    max_size: int = 8,
    fill_value: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply random cutout on one image (HWC). Useful as preprocessing_function in ImageDataGenerator."""
    if max_holes < 1:
        raise ValueError("max_holes must be >= 1")
    if min_size < 1 or max_size < min_size:
        raise ValueError("Invalid min_size/max_size values")

    local_rng = rng or np.random.default_rng()
    output = image.copy()

    height, width = output.shape[:2]
    max_allowed = max(1, min(max_size, height, width))
    min_allowed = max(1, min(min_size, max_allowed))

    for _ in range(max_holes):
        hole_size = int(local_rng.integers(min_allowed, max_allowed + 1))
        if hole_size >= height or hole_size >= width:
            output[...] = fill_value
            continue

        y = int(local_rng.integers(0, height - hole_size + 1))
        x = int(local_rng.integers(0, width - hole_size + 1))
        output[y : y + hole_size, x : x + hole_size, ...] = fill_value

    return output


def _as_label_indices(labels: Optional[ArrayLike]) -> Optional[np.ndarray]:
    if labels is None:
        return None
    return ensure_label_indices(labels)


def _predict_and_resolve_labels(
    model: Any,
    data: Any,
    labels: Optional[ArrayLike] = None,
    batch_size: Optional[int] = None,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if hasattr(data, "reset"):
        data.reset()

    predict_kwargs: Dict[str, Any] = {"verbose": verbose}
    if batch_size is not None:
        predict_kwargs["batch_size"] = batch_size

    y_pred_probs = model.predict(data, **predict_kwargs)
    y_pred = np.argmax(y_pred_probs, axis=-1)

    if labels is not None:
        y_true = _as_label_indices(labels)
    elif hasattr(data, "y"):
        y_true = _as_label_indices(data.y)
    else:
        raise ValueError(
            "Unable to infer ground-truth labels. Provide `labels` or pass a generator with `.y`."
        )

    return y_true, y_pred


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute accuracy, precision, recall and f1 for classification tasks."""
    y_true_idx = ensure_label_indices(y_true)
    y_pred_idx = ensure_label_indices(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_idx,
        y_pred_idx,
        average=average,
        zero_division=0,
    )

    if np.isscalar(support) or support is None:
        support_value = int(len(y_true_idx))
    else:
        support_value = int(np.sum(support))

    return {
        "accuracy": float(accuracy_score(y_true_idx, y_pred_idx)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": support_value,
    }


def predict_and_resolve_labels(
    model: Any,
    data: Any,
    labels: Optional[ArrayLike] = None,
    batch_size: Optional[int] = None,
    verbose: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Public wrapper that returns `(y_true, y_pred)` label indices."""
    return _predict_and_resolve_labels(
        model=model,
        data=data,
        labels=labels,
        batch_size=batch_size,
        verbose=verbose,
    )


def compute_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix from labels (supports one-hot input)."""
    y_true_idx = ensure_label_indices(y_true)
    y_pred_idx = ensure_label_indices(y_pred)
    return confusion_matrix(y_true_idx, y_pred_idx, normalize=normalize)


def build_classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    target_names: Optional[Sequence[str]] = None,
    digits: int = 4,
) -> str:
    """Build a text classification report with precision/recall/f1/support."""
    y_true_idx = ensure_label_indices(y_true)
    y_pred_idx = ensure_label_indices(y_pred)
    return classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=list(target_names) if target_names is not None else None,
        digits=digits,
        zero_division=0,
    )


def extract_final_history_metrics(history: Any) -> Dict[str, float]:
    """Extract final values from a Keras History object or a dict-like history."""
    source = getattr(history, "history", history)
    if not isinstance(source, Mapping):
        return {}

    final_metrics: Dict[str, float] = {}
    for metric_name, values in source.items():
        if isinstance(values, (list, tuple)) and values:
            final_metrics[f"final_{metric_name}"] = float(values[-1])
    return final_metrics


def log_metrics_to_csv(
    csv_path: Union[str, Path],
    metrics: Mapping[str, Any],
    append: bool = True,
) -> pd.DataFrame:
    """Write metrics to CSV while preserving a consistent canonical schema."""
    row = dict(metrics)

    for column in PERFORMANCE_REPORT_COLUMNS:
        row.setdefault(column, None)

    frame = pd.DataFrame([row])

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if append and csv_path.exists():
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
    else:
        combined = frame

    ordered_columns = [
        *[column for column in PERFORMANCE_REPORT_COLUMNS if column in combined.columns],
        *[column for column in combined.columns if column not in PERFORMANCE_REPORT_COLUMNS],
    ]
    combined = combined.reindex(columns=ordered_columns)
    combined.to_csv(csv_path, index=False)
    return combined.tail(1)


def _resolve_current_learning_rate(model: Any) -> Optional[float]:
    optimizer = getattr(model, "optimizer", None)
    if optimizer is None:
        return None

    learning_rate = getattr(optimizer, "learning_rate", None)
    if learning_rate is None:
        learning_rate = getattr(optimizer, "lr", None)
    if learning_rate is None:
        return None

    try:
        import tensorflow as tf

        return float(tf.keras.backend.get_value(learning_rate))
    except Exception:
        try:
            return float(learning_rate)
        except Exception:
            return None


def _resolve_test_loss(
    model: Any,
    data: Any,
    batch_size: Optional[int] = None,
    verbose: int = 0,
) -> Optional[float]:
    if hasattr(data, "reset"):
        data.reset()

    evaluate_kwargs: Dict[str, Any] = {"verbose": verbose}
    if batch_size is not None:
        evaluate_kwargs["batch_size"] = batch_size

    try:
        evaluation = model.evaluate(data, **evaluate_kwargs)
    except Exception:
        return None
    finally:
        if hasattr(data, "reset"):
            data.reset()

    if isinstance(evaluation, (list, tuple, np.ndarray)):
        if len(evaluation) == 0:
            return None
        return float(evaluation[0])

    try:
        return float(evaluation)
    except Exception:
        return None


def evaluate_and_log_model(
    model: Any,
    model_name: str,
    data: Any,
    csv_path: Union[str, Path],
    labels: Optional[ArrayLike] = None,
    history: Optional[Any] = None,
    train_time_sec: Optional[float] = None,
    average: str = "macro",
    batch_size: Optional[int] = None,
    verbose: int = 0,
    extra_fields: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a model and append a flexible performance row to CSV.

    Works with:
    - Numpy arrays + labels
    - tf.keras generators/iterators that expose `.y`
    - Any model supporting `.predict(...)`
    """
    y_true, y_pred = _predict_and_resolve_labels(
        model=model,
        data=data,
        labels=labels,
        batch_size=batch_size,
        verbose=verbose,
    )

    metrics = compute_classification_metrics(y_true, y_pred, average=average)

    normalized_train_time_sec = round(float(train_time_sec), 2) if train_time_sec is not None else None

    row: Dict[str, Any] = {
        "model_name": model_name,
        "num_params": int(model.count_params()) if hasattr(model, "count_params") else None,
        "train_time_sec": normalized_train_time_sec,
        "train_time": format_duration(normalized_train_time_sec),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        **metrics,
    }

    if history is not None:
        row.update(extract_final_history_metrics(history))

        source = getattr(history, "history", history)
        if isinstance(source, Mapping) and "accuracy" in source and isinstance(source["accuracy"], (list, tuple)):
            row["epochs_trained"] = len(source["accuracy"])

    if extra_fields:
        row.update(dict(extra_fields))

    if row.get("final_learning_rate") is None:
        row["final_learning_rate"] = _resolve_current_learning_rate(model)

    if row.get("test_loss") is None:
        row["test_loss"] = _resolve_test_loss(
            model=model,
            data=data,
            batch_size=batch_size,
            verbose=verbose,
        )

    log_metrics_to_csv(csv_path=csv_path, metrics=row, append=True)
    return row


def build_tf_dataset(
    x: ArrayLike,
    y: ArrayLike,
    batch_size: int,
    image_size: Optional[Tuple[int, int]] = None,
    shuffle: bool = False,
    shuffle_buffer: int = 10_000,
    augment_fn: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    preprocess_fn: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
    autotune: Optional[Any] = None,
):
    """Build a tf.data.Dataset pipeline with optional resize, preprocess and augmentation."""
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        ds = ds.shuffle(shuffle_buffer)

    if image_size is not None:
        target_h, target_w = image_size

        def _resize_fn(image, label):
            return tf.image.resize(image, (target_h, target_w)), label

        ds = ds.map(_resize_fn, num_parallel_calls=autotune or tf.data.AUTOTUNE)

    if preprocess_fn is not None:
        ds = ds.map(preprocess_fn, num_parallel_calls=autotune or tf.data.AUTOTUNE)

    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=autotune or tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(autotune or tf.data.AUTOTUNE)
    return ds


def make_run_log_dir(base_dir: Union[str, Path], prefix: str = "run") -> str:
    """Create and return a timestamped log directory path."""
    log_dir = Path(base_dir) / "logs" / f"{prefix}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return os.fspath(log_dir)


def apply_notebook01_plot_style(
    fig: Any,
    ax: Any,
    facecolor: str = "#212529",
    tick_color: str = "white",
    hide_spines: bool = True,
    grid_axis: Optional[str] = "y",
    grid_color: str = "gray",
    grid_alpha: float = 0.3,
) -> None:
    """Apply the dark visual style used in notebook 01 to one axes object."""
    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.tick_params(axis="x", colors=tick_color)
    ax.tick_params(axis="y", colors=tick_color)
    ax.xaxis.label.set_color(tick_color)
    ax.yaxis.label.set_color(tick_color)
    ax.title.set_color(tick_color)

    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if grid_axis is not None:
        ax.grid(axis=grid_axis, linestyle="--", alpha=grid_alpha, color=grid_color)


def plot_confusion_matrix_notebook01(
    confusion: np.ndarray,
    class_names: Sequence[str],
    title: str,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """Plot confusion matrix with notebook 01 dark style."""
    sns.set_theme(style="dark")
    fig, ax = plt.subplots(figsize=figsize)
    apply_notebook01_plot_style(fig, ax, grid_axis=None)

    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    plt.show()


def plot_training_curves_notebook01(
    history: Any,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300,
) -> None:
    """Plot accuracy and loss curves with notebook 01 dark style."""
    source = getattr(history, "history", history)
    train_acc = source.get("accuracy", []) if isinstance(source, Mapping) else []
    val_acc = source.get("val_accuracy", []) if isinstance(source, Mapping) else []
    train_loss = source.get("loss", []) if isinstance(source, Mapping) else []
    val_loss = source.get("val_loss", []) if isinstance(source, Mapping) else []

    sns.set_theme(style="dark")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for axis in axes:
        apply_notebook01_plot_style(fig, axis)

    axes[0].plot(train_acc, label="Train", color="#80ed99", linewidth=2)
    axes[0].plot(val_acc, label="Validation", color="#56cfe1", linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    legend0 = axes[0].legend()
    if legend0 is not None:
        for text in legend0.get_texts():
            text.set_color("white")

    axes[1].plot(train_loss, label="Train", color="#ff8fab", linewidth=2)
    axes[1].plot(val_loss, label="Validation", color="#f4a261", linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    legend1 = axes[1].legend()
    if legend1 is not None:
        for text in legend1.get_texts():
            text.set_color("white")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
