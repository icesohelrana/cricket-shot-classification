from torchmetrics import Accuracy,AveragePrecision,BinnedPrecisionRecallCurve,ConfusionMatrix
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

def metrics_init(num_classes=2):
    acc=Accuracy()
    avg_prec=AveragePrecision(num_classes)
    confusion_m = ConfusionMatrix(num_classes)
    metrices = {'acc':acc,'avg_prec':avg_prec,'confusion_m':confusion_m}
    return metrices

def metrics_op(metrics,preds,gts):
    for key in metrics:
        metrics[key](preds,gts)
    return
def metrics_compute(metrics):
    total_res=metrics
    for key in metrics:
        total_res[key] = metrics[key].compute()
        metrics[key].reset()
    return total_res
def print_metrics(metrics):
    for key in metrics:
        print(f'{key} : {metrics[key]}')
def tensorboard_metric_plot(writer,epoch,metrics_res,losses):
    writer.add_scaler('losses',epoch,losses)
    for key in metrics_res:
        if key=='confusion_m':
            writer.add_image(key,epoch,metrics_res[key])
        else: 
            writer.add_scaler(key,epoch,metrics_res[key])
    
