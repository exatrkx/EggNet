import torch


def get_optimizers(parameters, hparams):
    """Get the optimizer and scheduler."""
    weight_decay = hparams.get("lr_weight_decay", 0.01)
    optimizer = [
        torch.optim.AdamW(
            parameters,
            lr=(hparams["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
            weight_decay=weight_decay,
        )
    ]

    if (
        "scheduler" not in hparams
        or hparams["scheduler"] is None
        or hparams["scheduler"] == "StepLR"
    ):
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=hparams["patience"],
                    gamma=hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    elif hparams["scheduler"] == "ReduceLROnPlateau":
        metric_mode = hparams.get("metric_mode", "min")
        metric_to_monitor = hparams.get("metric_to_monitor", "val_loss")
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    mode=metric_mode,
                    factor=hparams["factor"],
                    patience=hparams["patience"],
                    verbose=True,
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": metric_to_monitor,
            }
        ]
    elif hparams["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer[0],
                    T_0=hparams["patience"],
                    T_mult=2,
                    eta_min=1e-8,
                    last_epoch=-1,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    else:
        raise ValueError(f"Unknown scheduler: {hparams['scheduler']}")

    return optimizer, scheduler
