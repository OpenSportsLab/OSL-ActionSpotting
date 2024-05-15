def get_default_args_train_e2e_dali(cfg):
    return {"classes": cfg.classes,
            'train': True,
            'acc_grad_iter': cfg.training.acc_grad_iter,
            'num_epochs': cfg.training.num_epochs,
            'repartitions': cfg.repartitions}


def get_default_args_val_e2e_dali(cfg):
    return {"classes": cfg.classes,
            'train': False,
            'acc_grad_iter': cfg.training.acc_grad_iter,
            'num_epochs': cfg.training.num_epochs,
            'repartitions': cfg.repartitions}


def get_default_args_train_e2e_opencv(cfg):
    return {"classes": cfg.classes,
            'train': True}


def get_default_args_val_e2e_opencv(cfg):
    return {"classes": cfg.classes,
            'train': False}


def get_default_args_train():
    return {'train': True}


def get_default_args_val():
    return {'train': False}


def get_default_args_val_data_frames_e2e_dali(cfg):
    return {"classes": cfg.classes,
            'repartitions': cfg.repartitions}


def get_default_args_val_data_frames_e2e_opencv(cfg):
    return {"classes": cfg.classes}


def get_default_args_dataset(split, cfg, e2e=False, dali=False):
    if split == 'train':
        if e2e:
            if dali:
                return get_default_args_train_e2e_dali(cfg)
            else:
                return get_default_args_train_e2e_opencv(cfg)
        else:
            return get_default_args_train()

    elif split == 'val':
        if e2e:
            if dali:
                return get_default_args_val_e2e_dali(cfg)
            else:
                return get_default_args_val_e2e_opencv(cfg)
        else:
            return get_default_args_val()

    elif split == 'val_data_frames' or split == 'test' or split == 'challenge':
        if e2e:
            if dali:
                return get_default_args_val_data_frames_e2e_dali(cfg)
            else:
                return get_default_args_val_data_frames_e2e_opencv(cfg)
        else:
            return
    else:
        return None


def get_default_args_model(cfg, e2e=False):
    if e2e:
        return {"classes": cfg.classes}
    else:
        return None


def get_default_args_trainer(cfg, e2e, dali, len_train_loader):
    if e2e:
        return {"len_train_loader": len_train_loader,
                'work_dir': cfg.work_dir,
                'dali': cfg.dali,
                'repartitions': cfg.repartitions if dali else None,
                'cfg_test': cfg.dataset.test,
                'cfg_challenge': cfg.dataset.challenge,
                'cfg_val_data_frames': cfg.dataset.val_data_frames
                }
    else:
        return None
