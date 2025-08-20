from typing import Optional, Union, Tuple

import torch
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss, ReLU
from transformers.modeling_outputs import SequenceClassifierOutput


def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None, output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
            base_model_output: Optional[torch.Tensor] = None) -> Union[
    SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
    if base_model_output is None:
        base_model_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    hidden_state = base_model_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = ReLU()(pooled_output)  # (bs, dim)
    pooled_output = self.dropout(pooled_output)  # (bs, dim)
    logits = self.classifier(pooled_output)  # (bs, num_labels)

    loss = None
    if labels is not None:
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"

        if self.config.problem_type == "regression":
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

    if not return_dict:
        output = (logits,) + base_model_output[1:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=base_model_output.hidden_states,
        attentions=base_model_output.attentions,
    )
