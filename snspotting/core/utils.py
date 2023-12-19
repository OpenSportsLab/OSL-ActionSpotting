import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import logging

class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, pl_module):
        # don't show the version number
        items = super().get_metrics(trainer,pl_module)
        items.pop("v_num", None)
        return items
    
class MyCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_validation_epoch_end(self, trainer, pl_module):
        loss_validation = pl_module.losses.avg
        state = {
                'epoch': trainer.current_epoch + 1,
                'state_dict': pl_module.model.state_dict(),
                'best_loss': pl_module.best_loss,
                'optimizer': pl_module.optimizer.state_dict(),
            }

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < pl_module.best_loss
        pl_module.best_loss = min(loss_validation, pl_module.best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            pl_module.best_state = state
            # torch.save(state, best_model_path)

        # Reduce LR on Plateau after patience reached
        prevLR = pl_module.optimizer.param_groups[0]['lr']
        pl_module.scheduler.step(loss_validation)
        currLR = pl_module.optimizer.param_groups[0]['lr']

        if (currLR is not prevLR and pl_module.scheduler.num_bad_epochs == 0):
            logging.info("\nPlateau Reached!")
        if (prevLR < 2 * pl_module.scheduler.eps and
            pl_module.scheduler.num_bad_epochs >= pl_module.scheduler.patience):
            logging.info("\nPlateau Reached and no more reduction -> Exiting Loop")
            trainer.should_stop=True