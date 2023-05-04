import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict
from .builder import CLASSIFIER, build_transformer
from mmcv.runner import BaseModule


def soft_cross_entropy(predicts, targets):
    student_likelihood = F.log_softmax(predicts, dim=-1)
    targets_prob = F.softmax(targets, dim=-1)
    return (-targets_prob * student_likelihood).mean()



@CLASSIFIER.register_module()
class DistillationClassifier(BaseModule):
    
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = build_transformer(teacher)
        self.student = build_transformer(student)
    
    def forward(self, img, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, **kwargs)
        else:
            return self.forward_test(img, **kwargs)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars
    
    def train_step(self, data, optimizer=None, **kwargs):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = data['img']
        student_logits, student_inter, student_value, student_query, student_key = \
            self.student(input_ids, segment_ids, input_mask)
        teacher_logits, teacher_inter, teacher_value, teacher_query, teacher_key = \
            self.teacher(input_ids, segment_ids, input_mask)
        
        
        cls_loss = soft_cross_entropy(student_logits, teacher_logits)
        gt_beta = 0.5
        logsoftmax_student_logits = F.log_softmax(student_logits, dim=-1)
        ce_loss = torch.nn.NLLLoss()
        tmp_loss = ce_loss(logsoftmax_student_logits, label_ids)
        cls_loss += gt_beta * tmp_loss
        
        losses = {
            'cls_loss': cls_loss
        }
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'][0]))

        return outputs
    
    def val_step(self, data, optimizer=None, **kwargs):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = data['img']
        student_logits, student_inter, student_value, student_query, student_key = \
            self.student(input_ids, segment_ids, input_mask)
        return student_logits
    
    def simple_test(self, data, **kwargs):
        input_ids, input_mask, segment_ids, label_ids, seq_lengths = data
        student_logits, student_inter, student_value, student_query, student_key = \
            self.student(input_ids, segment_ids, input_mask)
        return student_logits
    
    def forward_test(self, img, **kwargs):
        return self.simple_test(img, **kwargs)
