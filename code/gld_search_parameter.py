import torch
from torch.nn import Parameter
from nonparametrics import dp_div
from option import Option
from training_arguments import num_option_levels, default_radius_scheduler, default_max_r, \
    default_sampling_size, default_sampling_steps, default_norm_weight


class GldSearchParameter:
    def __init__(self, parameter: Parameter, max_r=default_max_r, sampling_size=default_sampling_size,
                 sampling_steps=default_sampling_steps, radius_scheduler=default_radius_scheduler,
                 norm_weight=default_norm_weight, sampling_idx: torch.Tensor = None):
        """
        A tensor to be trained by a GldSearchTrainer

        :param parameter: The tensor to train
        :param max_r: Maximal search radius for this tensor
        :param sampling_size: The number of parameters to train from this tensor per iteration
        :param sampling_steps: The number of training iterations after which this tensor will reselect parameters
            to train
        :param radius_scheduler: A scheduler function to return a multiplier for this tensor's maximal radius
            each iteration
        :param norm_weight: The weight of this tensor's L2-norm when calculating a regularized objective
        :param sampling_idx: A list of fixed indices of parameters from this tensor train. Has no effect on
            the subclasses of GldSearchParameter as they implement functionality to select the indices themselves.
        """
        self.parameter = parameter
        self.max_r = max_r
        self.sampling_size = sampling_size if sampling_size > 0 else self.parameter.numel()
        self.sampling_steps = sampling_steps
        self.radius_scheduler = radius_scheduler
        self.norm_weight = norm_weight
        self.sampling_idx = sampling_idx

        self.steps = 0
        self.radius = max_r
        option = Option()
        self.options = [option]
        for i in range(num_option_levels - 1):
            option.options.append(Option())
            option = option.options[0]

    def step(self):
        self.steps += 1
        if self.radius_scheduler is not None:
            self.radius = self.max_r * self.radius_scheduler(self.steps)

    def sample(self, model, inputs, original_outputs):
        pass

    def perturb(self, k):
        r = self.radius / 2 ** k
        perturb_idx = self.sampling_idx[torch.arange(self.sampling_size)]
        perturbation = r * torch.normal(0, 1, size=(self.sampling_size,), device=self.parameter.device)
        self.parameter.flatten()[perturb_idx] += perturbation
        return perturbation, perturb_idx

    def calculate_norm(self):
        return self.parameter.norm() * self.norm_weight


class AllParameter(GldSearchParameter):
    def perturb(self, k):
        r = self.radius / 2 ** k
        perturbation = r * torch.normal(0, 1, size=self.parameter.shape, device=self.parameter.device)
        self.parameter += perturbation
        return perturbation.flatten(), torch.arange(self.parameter.numel())


class RandomParameter(GldSearchParameter):
    def sample(self, model, inputs, original_outputs):
        if self.steps % self.sampling_steps == 0:
            self.sampling_idx = torch.randperm(self.parameter.numel())


class AbsParameter(GldSearchParameter):
    def sample(self, model, inputs, original_outputs):
        if self.steps % self.sampling_steps == 0:
            self.sampling_idx = self.parameter.flatten().abs().sort(descending=True)[1]


class FisherParameter(GldSearchParameter):
    def sample(self, model, inputs, original_outputs):
        if self.steps % self.sampling_steps == 0:
            d = torch.zeros(self.parameter.numel())
            std = torch.max(torch.abs(self.parameter)).item()
            for divergence, parameter in zip(d, self.parameter.flatten()):
                perturbation = torch.normal(0, std, size=())
                parameter += perturbation
                outputs = model(**inputs)
                divergence += dp_div(original_outputs[1].cpu(), outputs[1].cpu())[0]
                parameter -= perturbation
            self.sampling_idx = d.sort(descending=True)[1]
