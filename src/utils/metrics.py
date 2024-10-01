from abc import ABC, abstractmethod

from . import calculate_flow_error

def log_best_metrics(best_metric_value, metric_key_prefix, metrics_to_log, trainer,
                     logger, target_metric=None):
    if trainer.global_rank != 0:
        return best_metric_value

    if target_metric is None:
        target_metric = f'{metric_key_prefix}/{metrics_to_log[0]}'

    accumulated_metric = trainer.callback_metrics.get(target_metric)
    if accumulated_metric is None or accumulated_metric >= best_metric_value:
        return best_metric_value

    best_metric_value = accumulated_metric
    log_summary = {}
    for metric in metrics_to_log:
        value = trainer.callback_metrics.get(f'{metric_key_prefix}/{metric}')
        if value is not None:
            metric_name = metric
            log_summary[f"{metric_name}_at_best"] = value

    try:
        logger.experiment.summary.update(log_summary)
    except:
        for key, value in log_summary.items():
            logger.experiment.add_scalar(key, value, global_step=trainer.global_step)

    return best_metric_value

class ErrorCalculatorFactory:
    @staticmethod
    def get_error_calculator(data_type):
        if data_type == 'DSEC' or data_type == 'MVSEC':
            return OpticalFlowError()
        else:
            raise ValueError("Unsupported dataset type")

class ErrorBase(ABC):
    @abstractmethod
    def run(predictions, batch):
        pass

    @abstractmethod
    def log_best(best_val, trainer, logger):
        pass

class OpticalFlowError(ErrorBase):
    @staticmethod
    def run(predictions, batch):
        flow_pred = predictions['flow']
        flow_gt = batch['forward_flow']
        gt_valid = batch['flow_valid']
        return calculate_flow_error(flow_gt, flow_pred, gt_valid)

    @staticmethod
    def log_best(best_EPE, trainer, logger, target_metric=None):
        metrics_to_log = ['EPE', 'AE', '3PE']
        return log_best_metrics(best_EPE, 'val_losses', metrics_to_log, trainer, logger,
                                target_metric)
