import os
import pickle
import torch
from mmcls.datasets.builder import DATASETS
from torch.utils.data import Dataset, TensorDataset
from .tokenization import BertTokenizer
import numpy as np


from .utils import (
    ColaProcessor,
    MnliProcessor,
    MnliMismatchedProcessor,
    MrpcProcessor,
    Sst2Processor,
    StsbProcessor,
    QqpProcessor,
    QnliProcessor,
    RteProcessor,
    convert_examples_to_features,
    compute_metrics
)


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification"
}


def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features],
                                     dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features],
                                     dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features],
                                   dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features],
                                  dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features],
                                   dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids


@DATASETS.register_module()
class GLUEDataset(Dataset):
    
    def __init__(self,
                 task,
                 data_prefix,
                 max_seq_length,
                 tokenizer_path,
                 aug_train=False,
                 test_mode=False):
        super().__init__()
        assert task in processors
        self.task = task
        self.processor = processors[task]()
        self.output_mode = output_modes[task]
        self.label_list = self.processor.get_labels()
        self.num_lables = len(self.label_list)
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=True)
        if not test_mode:
            if aug_train:
                self.data_examples = self.processor.get_aug_examples(data_prefix)
            else:
                self.data_examples = self.processor.get_train_examples(data_prefix)
        else:
            self.data_examples = self.processor.get_dev_examples(data_prefix)
        self.data_features = convert_examples_to_features(
            self.data_examples, self.label_list, max_seq_length,
            self.tokenizer, self.output_mode)
        self.data, self.labels = get_tensor_data(self.output_mode, self.data_features)
        self.labels = self.labels.numpy()
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = {'img': self.data[idx]}
        return data

    def evaluate(self, results, logger=None):
        preds = torch.stack(results, dim=0)
        preds = preds.cpu().detach().numpy()
        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.squeeze(preds)
        output = compute_metrics(self.task, preds, self.labels)
        return output