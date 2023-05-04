from typing import Optional, Union
from pathlib import Path
import os
import random

import torch
import torch.utils.data as data
import torchaudio

import numpy as np

from mmcls.datasets.builder import DATASETS
from mmcls.datasets.pipelines import Compose
from mmcls.models.losses import accuracy
from mmcls.core.evaluation import precision_recall_f1, support
# from mmcls.datasets.base_dataset import BaseDataset

NOISE_FOLDER = "_background_noise_"

@DATASETS.register_module()
class SpeechCommandDataset(data.Dataset):
    def __init__(
        self,
        data_prefix: Union[str, Path],
        folder_in_archive: str = "SpeechCommands",
        download: bool = False,
        subset: Optional[str] = None,
        silence_percent=0.1,
        pipeline=None,
        noise_ratio=None,
        noise_max_scale=0.4,
        cache_origin_data=False,
        test_mode=True,
        num_classes=12,
        version = "speech_commands_v0.01"
    ) -> None:
        self.classes_12 = ['unknown', 'silence', 'yes', 'no', 'up', 'down', 'left', 'right',
                           'on', 'off', 'stop', 'go'
        ]
        self.classes_20 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 
                           'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'] 
        self.classes_35 = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
                           'four', 'go', 'happy', 'house', 'learn', 'left',  'marvin', 'nine', 'no', 'off', 
                           'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
                           'up', 'visual', 'wow', 'yes', 'zero'] 
        self.CLASSES = eval('self.classes_%d' % num_classes)
        dataset = torchaudio.datasets.SPEECHCOMMANDS(data_prefix, version,
                                                     folder_in_archive,
                                                     download, subset)
        data_path = os.path.join(data_prefix, folder_in_archive, version)
        self.num_classes = num_classes
        self.datas = list()
        for fileid in dataset._walker:
            relpath = os.path.relpath(fileid, data_path)
            label, _ = os.path.split(relpath)
            label = self.name_to_label(label)
            if (label == -1):
                continue
            self.datas.append([fileid, label])

        self.sample_rate = 16000

        # setup silence
        if silence_percent > 0 and num_classes == 12:
            silence_data = [['', self.name_to_label('silence')]
                            for _ in range(int(len(dataset) * silence_percent))]
            self.datas.extend(silence_data)

        # setup noise
        self.noise_folder = os.path.join(data_prefix, folder_in_archive, version,
                                         NOISE_FOLDER)
        self.noise_files = sorted(str(p) for p in Path(self.noise_folder).glob('*.wav')) \
            if subset == 'training' and noise_ratio != None else None
        
        self.pipeline = Compose(pipeline)
        self.noise_ratio = noise_ratio
        self.noise_max_scale = noise_max_scale
        self.silence_ratio = silence_percent
        if noise_ratio is not None and subset is 'training':
            assert 0 < noise_max_scale < 1
        assert num_classes == 12 or num_classes == 20 or num_classes == 35, 'only support 12/20//35 now'
        self.cache_origin = cache_origin_data
        self.origin_datas = dict()
        self.origin_noise_datas = dict()

    def __len__(self):
        return len(self.datas)

    def label_to_name(self, label): # useless function
        if label >= 12:
            return 'unknown'
        return self.classes_12[label]

    def name_to_label(self, name): 
        if self.num_classes == 12:
            if name not in self.classes_12:
                return 0
            return self.classes_12.index(name)
        elif self.num_classes == 20:
            if name not in self.classes_20:
                return 0 if self.classes_20 == 'unknown' else -1
            return self.classes_20.index(name)
        elif self.num_classes == 35:
            if name not in self.classes_35:
                return 0 if self.classes_35 == 'unknown' else -1
            return self.classes_35.index(name)
        else:
            raise RuntimeError

    def __getitem__(self, index):
        """
        return feature and label
        """
        # Tensor, int, str, str, int
        if index in self.origin_datas.keys():
            [waveform, _, label] = self.origin_datas[index]
        else:
            waveform, sample_rate, label = self.pull_origin(index)
            if sample_rate != self.sample_rate:
                raise RuntimeError
            if self.cache_origin:
                self.origin_datas[index] = [waveform, sample_rate, label]

        if self.noise_files is not None and random.uniform(
                0, 1) < self.noise_ratio:
            noise_file = random.choice(self.noise_files)
            if noise_file in self.origin_noise_datas.keys():
                waveform_noise = self.origin_noise_datas[noise_file]
            else:
                waveform_noise, _ = torchaudio.load(noise_file)
                if self.cache_origin:
                    self.origin_noise_datas[noise_file] = waveform_noise
            noise_len = waveform_noise.shape[1]
            wav_len = waveform.shape[1]
            if noise_len >= wav_len:
                rand_start = random.randint(0, noise_len - wav_len - 1)
                waveform_noise = waveform_noise[:,
                                                rand_start:wav_len + rand_start]
            else:
                waveform_noise = torch.nn.functional.pad(
                    waveform_noise, (0, wav_len - noise_len))
            random_scale = random.uniform(0, self.noise_max_scale)
            waveform = waveform * (1 -
                                   random_scale) + waveform_noise * random_scale

        data = {'img': waveform, 'gt_label': label}
        return self.pipeline(data)

    def pull_origin(self, index):
        """
        get original item
        """
        [data_id, label] = self.datas[index]
        if data_id != '':
            waveform, sample_rate = torchaudio.load(data_id)
        else:
            waveform = torch.zeros(1, 16000)
            sample_rate = 16000
        return waveform, sample_rate, label

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """
        if not hasattr(self, 'gt_label'):
            self.gt_labels = np.array([label for data_id, label in self.datas])
        
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
            metric_options = {'topk': (1, 5)}
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
