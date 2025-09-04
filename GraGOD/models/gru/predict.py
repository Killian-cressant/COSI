import argparse
import json
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.config import get_dataset_config
from datasets.dataset import SlidingWindowDataset
from gragod import InterPolationMethods, ParamFileTypes
from gragod.metrics.calculator import get_metrics
from gragod.metrics.visualization import print_all_metrics
from gragod.predictions.prediction import generate_scores, get_threshold
from gragod.training import load_params, load_training_data, set_seeds
from gragod.types import CleanMethods, cast_dataset
from models.gru.model import GRU_PLModule, GRUModel

RANDOM_SEED = 42
set_seeds(RANDOM_SEED)


def run_model(
    model: GRU_PLModule,
    loader: DataLoader,
    device: str,
) -> torch.Tensor:
    """
    Generate predictions for a given loader.

    Args:
        model: The model to use for prediction.
        loader: The loader to use for prediction.
        device: The device to use for prediction.

    Returns:
        torch.Tensor: The predictions.
    """
    trainer = pl.Trainer(accelerator=device)
    output = trainer.predict(model, loader)
    forecasts = torch.cat(output)  # type: ignore

    return forecasts


def main(
    dataset_name: str,
    model_params: dict,
    batch_size: int = 264,
    ckpt_path: str | None = None,
    device: str = "mps",
    n_workers: int = 0,
    target_dims: int | None = None,
    save_dir: str = "output",
    test_size: float = 0.1,
    val_size: float = 0.1,
    clean: str = "interpolate",
    interpolate_method: InterPolationMethods | None = None,
    params: dict = {},
    **kwargs,
) -> dict:
    """
    Main function to load data, model and generate predictions.

    Returns:
        dict: A dictionary containing predictions, labels, scores, data, thresholds,
        forecasts, reconstructions, and metrics.
    """
    return_dict = {}

    checkpoint_path = (
        os.path.join(params["predictor_params"]["ckpt_folder"], "gru.ckpt")
        if ckpt_path is None
        else ckpt_path
    )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")

    dataset = cast_dataset(dataset_name)
    dataset_config = get_dataset_config(dataset=dataset)

    # Load data
    (
        X_train,
        X_val,
        X_test,
        X_train_labels,
        X_val_labels,
        X_test_labels,
    ) = load_training_data(
        dataset=dataset,
        test_size=test_size,
        val_size=val_size,
        normalize=dataset_config.normalize,
        clean=clean == CleanMethods.INTERPOLATE,
        interpolate_method=interpolate_method,
    )

    window_size = model_params["window_size"]

    X_train_labels = X_train_labels[window_size:]
    X_val_labels = X_val_labels[window_size:]
    X_test_labels = X_test_labels[window_size:]

    # Create test dataloader
    train_dataset = SlidingWindowDataset(
        X_train,
        window_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )
    val_dataset = SlidingWindowDataset(X_val, window_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )
    test_dataset = SlidingWindowDataset(X_test, window_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
    )

    # Create and load model
    n_features = X_train.shape[1]
    model = GRUModel(
        n_features=n_features,
        **model_params,
    )

    print(f"Loading model from checkpoint: {checkpoint_path}")
    lightning_module = GRU_PLModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        model_params=model_params,
        **params["train_params"],
    )

    model = lightning_module.model.to(device)
    model.eval()

    # Generate predictions and calculate metrics
    forecasts_val = run_model(
        model=lightning_module,
        loader=val_loader,
        device=device,
    )
    forecasts_test = run_model(
        model=lightning_module,
        loader=test_loader,
        device=device,
    )

    val_scores = generate_scores(
        predictions=forecasts_val,
        true_values=X_val[window_size:],
        score_type=params["predictor_params"]["score_type"],
        post_process=params["predictor_params"]["post_process_scores"],
        window_size_smooth=params["predictor_params"]["window_size_smooth"],
    )
    test_scores = generate_scores(
        predictions=forecasts_test,
        true_values=X_test[window_size:],
        score_type=params["predictor_params"]["score_type"],
        post_process=params["predictor_params"]["post_process_scores"],
        window_size_smooth=params["predictor_params"]["window_size_smooth"],
    )

    thresholds = get_threshold(
        dataset=dataset,
        scores=val_scores,
        labels=X_val_labels,
        n_thresholds=params["predictor_params"]["n_thresholds"],
    )

    if torch.any(X_train_labels == 1):
        forecasts_train = run_model(
            model=lightning_module,
            loader=train_loader,
            device=device,
        )
        train_scores = generate_scores(
            predictions=forecasts_train,
            true_values=X_train[window_size:],
            score_type=params["predictor_params"]["score_type"],
            post_process=params["predictor_params"]["post_process_scores"],
            window_size_smooth=params["predictor_params"]["window_size_smooth"],
        )

        X_train_pred = (train_scores > thresholds).float()
        train_metrics = get_metrics(
            dataset=dataset,
            predictions=X_train_pred,
            labels=X_train_labels,
            scores=train_scores,
        )
        print_all_metrics(train_metrics, "------- Train -------")
        json.dump(
            train_metrics,
            open(
                os.path.join(
                    params["predictor_params"]["ckpt_folder"], "train_metrics.json"
                ),
                "w",
            ),
        )

        return_dict["train"] = {
            "predictions": X_train_pred,
            "labels": X_train_labels,
            "scores": train_scores,
            "data": X_train,
            "thresholds": thresholds,
            "forecasts": forecasts_train,
            "metrics": train_metrics,
        }

    X_val_pred = (val_scores > thresholds).float()
    metrics_val = get_metrics(
        dataset=dataset,
        predictions=X_val_pred,
        labels=X_val_labels,
        scores=val_scores,
    )
    print_all_metrics(metrics_val, "------- Val -------")

    X_test_pred = (test_scores > thresholds).float()
    metrics_test = get_metrics(
        dataset=dataset,
        predictions=X_test_pred,
        labels=X_test_labels,
        scores=test_scores,
    )
    print_all_metrics(metrics_test, "------- Test -------")

    # save
    json.dump(
        metrics_val,
        open(
            os.path.join(params["predictor_params"]["ckpt_folder"], "metrics_val.json"),
            "w",
        ),
    )
    return {
        "val": {
            "predictions": X_val_pred,
            "labels": X_val_labels,
            "scores": val_scores,
            "data": X_val,
            "thresholds": thresholds,
            "forecasts": forecasts_val,
            "metrics": metrics_val,
        },
        "test": {
            "predictions": X_test_pred,
            "labels": X_test_labels,
            "scores": test_scores,
            "data": X_test,
            "thresholds": thresholds,
            "forecasts": forecasts_test,
            "metrics": metrics_test,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="models/gru/params.yaml")
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    params = load_params(args.params_file, file_type=ParamFileTypes.YAML)

    main(
        dataset_name=params["dataset"],
        ckpt_path=args.ckpt_path,
        **params["train_params"],
        model_params=params["model_params"],
        params=params,
    )
