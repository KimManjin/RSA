import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import fvcore.nn as fv
import ops.logging as logging

logger = logging.get_logger(__name__)

def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))

        print((grad_out[0]))
        print((grad_in[0]))

    return hook


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def log_add(log_a, log_b):
    return log_a + np.log(1 + np.exp(log_b - log_a))


def class_accuracy(prediction, label):
    cf = confusion_matrix(prediction, label)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    cls_acc = cls_hit / cls_cnt.astype(float)

    mean_cls_acc = cls_acc.mean()

    return cls_acc, mean_cls_acc

def get_FLOPs_params(model, arch, x):
    def _human_format(num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        # add more suffixes if you need them
        return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    
    out = fv.FlopCountAnalysis(model, x)
    params = fv.parameter_count(model)
    params_table = fv.parameter_count_table(model)
    flops_human = _human_format(out.total())
    params_human =  _human_format(int(params['']))
    
    logger.info('\n')
    logger.info('-'*50)
    logger.info('Models\tFrames\tFLOPs\tParams')
    logger.info('-'*50)
    logger.info('%s\t%d\t%s\t%s' % (arch, x.size(0), flops_human,  params_human))
    logger.info('-'*50)
    logger.info('\n')
    
    return out.total(), int(params[''])