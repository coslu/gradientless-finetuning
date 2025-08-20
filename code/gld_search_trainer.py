import torch
from typing import Dict, Union, Any
from transformers import Trainer
from gld_search_parameter import GldSearchParameter
from option import Option
from training_arguments import num_options, num_search_radii


def copy(param):
    return param.data.clone().detach()


class GldSearchTrainer(Trainer):
    def __init__(self, trainable_params=None, **kwargs):
        super().__init__(**kwargs)
        self.trainable_params = trainable_params if trainable_params is not None else [GldSearchParameter(p) for p in
                                                                                       self.model.parameters()]
        for param in self.model.parameters():
            param.requires_grad = False

    def compute_option_loss(self, option: Option, param: GldSearchParameter, model, inputs):
        original_param = copy(param.parameter)
        option.apply(param.parameter)
        option_loss = float('inf')

        if len(option.options) == 0:
            changed_param = copy(param.parameter)  # after applying the current option

            # The option to not change the current option
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) + param.calculate_norm()
            option_loss = min(option_loss, loss)
            option.options.append(Option(loss))

            for k in range(num_search_radii):
                perturbation, perturb_idx = param.perturb(k)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs) + param.calculate_norm()
                option_loss = min(option_loss, loss)
                option.options.append(Option(loss, perturbation, perturb_idx))
                param.parameter.data = copy(changed_param)

            option.options.sort()
            option.options = option.options[:num_options]
        else:
            for child in option.options:
                option_loss = min(option_loss, self.compute_option_loss(child, param, model, inputs))

        param.parameter.data = copy(original_param)
        return option_loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        inputs |= {'base_model_output': model.base_model(**{k: inputs[k] for k in inputs if k != 'labels'})}

        for param in self.trainable_params:
            with self.compute_loss_context_manager():
                current_loss, original_outputs = self.compute_loss(model, inputs, return_outputs=True)
            param.sample(model, inputs, original_outputs)
            param.step()
            min_param = copy(param.parameter)  # Parameter will be updated to this after this training step
            min_loss_param = current_loss + param.calculate_norm()  # Minimum loss for the current parameter across all options
            next_options = param.options[0].options  # param.options will be set to this after this training step

            for option in param.options:
                loss = self.compute_option_loss(option, param, model, inputs)
                # This option is to be picked at the end if one of its perturbations gives the best loss so far
                if loss < min_loss_param:
                    min_loss_param = loss
                    min_param = copy(param.parameter)
                    option.apply(min_param)
                    next_options = option.options

            # Finally update the parameter
            param.parameter.data = copy(min_param)
            param.options = next_options

        return current_loss
