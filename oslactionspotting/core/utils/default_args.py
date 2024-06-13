def get_default_args_data_train_e2e_dali(cfg):
    return {
        "classes": cfg.classes,
        "train": True,
        "acc_grad_iter": cfg.training.acc_grad_iter,
        "num_epochs": cfg.training.num_epochs,
        "repartitions": cfg.repartitions,
    }


def get_default_args_data_valid_e2e_dali(cfg):
    return {
        "classes": cfg.classes,
        "train": False,
        "acc_grad_iter": cfg.training.acc_grad_iter,
        "num_epochs": cfg.training.num_epochs,
        "repartitions": cfg.repartitions,
    }


def get_default_args_data_train_e2e_opencv(cfg):
    return {"classes": cfg.classes, "train": True}


def get_default_args_data_valid_e2e_opencv(cfg):
    return {"classes": cfg.classes, "train": False}


def get_default_args_data_train():
    return {"train": True}


def get_default_args_data_valid():
    return {"train": False}


def get_default_args_data_valid_data_frames_e2e_dali(cfg):
    return {"classes": cfg.classes, "repartitions": cfg.repartitions}


def get_default_args_data_valid_data_frames_e2e_opencv(cfg):
    return {"classes": cfg.classes}


def get_default_args_dataset(split, cfg):
    if split == "train":
        if cfg.runner.type == "runner_e2e":
            if getattr(cfg, "dali", False):
                return get_default_args_data_train_e2e_dali(cfg)
            else:
                return get_default_args_data_train_e2e_opencv(cfg)
        else:
            return get_default_args_data_train()

    elif split == "valid":
        if cfg.runner.type == "runner_e2e":
            if getattr(cfg, "dali", False):
                return get_default_args_data_valid_e2e_dali(cfg)
            else:
                return get_default_args_data_valid_e2e_opencv(cfg)
        else:
            return get_default_args_data_valid()

    elif split == "valid_data_frames" or split == "test" or split == "challenge":
        if cfg.runner.type == "runner_e2e":
            if getattr(cfg, "dali", False):
                return get_default_args_data_valid_data_frames_e2e_dali(cfg)
            else:
                return get_default_args_data_valid_data_frames_e2e_opencv(cfg)
        else:
            return
    else:
        return None


def get_default_args_model(cfg):
    if cfg.model.type == "E2E":
        return {"classes": cfg.classes}
    else:
        return None


def get_default_args_trainer(cfg, len_train_loader):
    if cfg.training.type == "trainer_e2e":
        return {
            "len_train_loader": len_train_loader,
            "work_dir": cfg.work_dir,
            "dali": cfg.dali,
            "repartitions": cfg.repartitions if cfg.dali else None,
            "cfg_test": cfg.dataset.test,
            "cfg_challenge": cfg.dataset.challenge,
            "cfg_valid_data_frames": cfg.dataset.valid_data_frames,
        }
    else:
        return {"work_dir": cfg.work_dir}


def get_default_args_train(model, train_loader, valid_loader, classes, trainer_type):
    if trainer_type == "trainer_CALF" or trainer_type == "trainer_pooling":
        return {
            "model": model,
            "train_dataloaders": train_loader,
            "val_dataloaders": valid_loader,
        }
    elif trainer_type == "trainer_e2e":
        return {
            "train_loader": train_loader,
            "valid_loader": valid_loader,
            "classes": classes,
        }
