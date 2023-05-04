from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose
from mmcls.models.losses import accuracy

import os
import torch
import numpy as np
from torch_geometric.datasets import ModelNet as PyGModelNet
import torch_geometric.transforms as T


def pc_normalize(pc):
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc * pc, dim=1)))
    pc = pc / m
    return pc


@DATASETS.register_module()
class ModelNet(PyGModelNet):
    def __init__(self, data_prefix, test_mode=False):
        # Default setting
        pre_transform = T.NormalizeScale()
        transform = T.SamplePoints(1024)
        pre_filter = None

        self.test_mode = test_mode
        data_path = os.path.join(data_prefix, 'ModelNet40')
        super().__init__(data_path, name='40', train=not test_mode,
                         transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        label = data.y.item()
        data = torch.cat((pc_normalize(data.pos), data.pos), dim=-1)
        if self.test_mode:
            output = {'img': data}
        else:
            output = {'img': data, 'gt_label': label}
        return output

    def get_gt_labels(self):
        labels = []
        self.test_mode = False
        if not hasattr(self, 'gt_label'):
            for data in self:
                labels.append(data['gt_label'])
        self.gt_labels = np.array(labels)
        self.test_mode = True
        return self.gt_labels

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
