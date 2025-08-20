from typing import Callable


# Implement additional radius schedulers of type (int) -> float,
# taking the current step and returning a multiplier for max_r
def linear_schedule(current_step: int, max_steps=80000, min_rate=0.1):
    return 1 - current_step * (1 - min_rate) / max_steps


# Change training arguments
num_option_levels = 1  # number (depth) of option levels
num_options = 1  # number of options to store in each level
num_search_radii = 5  # number of perturbations to try at each new level, each halving the radius, starting from max_r

# Change the default values for the attributes of GldSearchParameter, can be set individually for each parameter
default_max_r = 7.5e-3
default_radius_scheduler: Callable[[int], float] | None = linear_schedule
default_sampling_size = -1
default_sampling_steps = 1
default_norm_weight = 0.05
