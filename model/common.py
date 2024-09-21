import copy
from typing import Dict, Optional

import numpy as np
from math import sqrt, floor
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback, EarlyStopping

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar

import torch.nn.functional as F
from load_dag import DAG

####################
# Collection of common building blocks for the models/training
####################


def block(n_hidden: int, n_layers: int):
    """Creates a fully-connected ``n_layers`` linear layers with ``n_hidden`` units"""

    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(0, inplace=True)]
    return layers


class GumbelMaxBinary(nn.Module):
    """Gumbel-max trick for binary variables"""
    # TODO temperature annealing
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, x):
        y_one_hot = F.gumbel_softmax(x, tau=self.tau, hard=True)
        return y_one_hot[:,0:1]
    
    
    
class GumbelMaxLayer(nn.Module):
    """Gumbel-max trick to sample from a categorical distribution. This is used to generate latent variables."""
    def __init__(self, tau, input_size):
        super().__init__()
        self.tau = tau
        self.input = nn.Parameter(torch.zeros(input_size))

    def forward(self):
        y_one_hot = F.gumbel_softmax(self.input, tau=self.tau, hard=True)
        return y_one_hot

# TODO: define latent distribution generator: ReLu + GumbelMaxLayer
class LatentGenerator(nn.Module):
    def __init__(self, latent_dim: int ,n_hidden: int, n_layers: int, n_components):
        super().__init__()
        self.gumbel_max = GumbelMaxLayer(tau = 0.1, input_size = n_components)
        n_hidden = n_hidden * n_components
        self.latent_dim = latent_dim
        self.model = nn.Sequential(nn.Linear(n_components, n_hidden), nn.LeakyReLU(0, inplace=True)
                                   , *block(n_hidden, n_layers - 1), nn.Linear(n_hidden, latent_dim * n_components))
    def forward(self, z):
        x = self.model(z)
        w = self.gumbel_max().view(-1)
        output = torch.zeros_like(x[:,0: self.latent_dim])
        for k in range(w.shape[0]):
            output += x[:, k*self.latent_dim: (k+1)*self.latent_dim] * w[k]
        return output
        
# Define monotonic network from paper Monotone and Partially Monotone Neural Networks
# The network has three hidden layers. The first layer is linear, then followed by group min and max.

class min_max_net(nn.Module):
    def __init__(self, input_dim: int, n_group: int, n_group_size: int, montn_index):
        super().__init__()
        self.input_dim = input_dim
        self.n_group_size = n_group_size
        self.n_group = n_group
        self.montn_index = montn_index
        self.linear = nn.Linear(input_dim, n_group_size * n_group)

    def forward(self, x):
        weight = self.linear.weight.data.clone()
        weight[:,self.montn_index] = weight[:,self.montn_index]**2  # Square the weights
        x = F.linear(x, weight, self.linear.bias)
        for i in range(self.n_group):
            x[:, i] = torch.min(x[:, i * self.n_group_size: (i + 1) * self.n_group_size], dim=1).values
        output = torch.max(x[:,0:self.n_group], dim=1).values
        for i in range(0,output.shape[0]):
            output[i] = (torch.rand(1, device=output.device) < torch.sigmoid(output[i])).float()
        return output.reshape(-1,1)

class monotonic_binary_output(nn.Module):
    # Input dimension: 2, binary ouput 
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear = nn.Linear(latent_dim,2)
        
    def forward(self,x):
        prop = torch.sigmoid(self.linear(x[:,:self.latent_dim]))
        z = x[:,self.latent_dim]
        output = torch.zeros_like(z)
        for i in range(0,output.shape[0]):
            if z[i] == 1:
                output[i] = (torch.rand(1, device=output.device) < (torch.max(prop[i,0:2]))).float()
            else:
                output[i] = (torch.rand(1, device=output.device) < (torch.min(prop[i,0:2]))).float()
        return output.reshape(-1,1)
            
            
# TODO: combine all networks network together.

class Generator(nn.Module):
    """The G-constrained generative model. For observational data, we sample from the noise and generate each variable
    according to the topological order of variables in the DAG. For interventional data, we assign the intervened
    variable to the given value and sample the rest of the variables according to the topological order.

    Args:
        dag (DAG): The causal graph.
        n_hidden (int): The number of hidden units in each layer of the generator.
        n_layers (int): The number of layers in the generator.

        n_hidden_latent (int): The number of hidden units in each layer of the latent generator.
        n_layers_latent (int): The number of layers in the latent generator.
    """
    def __init__(self, dag: DAG, upper_bound: Optional[Dict[int, torch.Tensor]],
                 lower_bound: Optional[Dict[int, torch.Tensor]], n_hidden: int, n_layers: int
                 , n_hidden_latent: int, n_layers_latent: int, latent_out_dim: int = 1):
        super().__init__()
        self.latent_dim = dag.latent_dim
        self.graph = dag.graph
        self.var_dims = dag.var_dims
        self.binary_keys = dag.binary_keys
        self.n_latent = dag.n_latent
        self.latent_out_dim = latent_out_dim
        if hasattr(dag, 'monotonic_index'):
            self.monotonic_index = dag.monotonic_index
        else:
            self.monotonic_index = {}
        
        self.upper = upper_bound
        self.lower = lower_bound

        self.model_dict = {}
        self.latent_gen_dict = {}

        # Create the model for each variable w.r.t. the topological order
        # Key are the variables and values are the input index of variables. Positive means observed and negative means latent.

        # create the latent generator for each latent variable
        for i in range(self.n_latent):
            # latent_dim =  n_components 
            self.latent_gen_dict[i] = LatentGenerator(latent_out_dim, n_hidden_latent, n_layers_latent, self.latent_dim)

        for key, value in self.graph.items():
            inputs = np.array(value).astype(int)
            if len(inputs) == 0:
                continue
            observed = inputs[inputs >= 0]
            latent = abs(inputs[inputs < 0])
            input_dim = len(latent) * latent_out_dim  + np.sum(self.var_dims[observed])
            if n_layers == 0:
                if self.binary_keys is None or key not in self.binary_keys:
                    last_layer = [nn.Linear(input_dim,  self.var_dims[key])]

                else:
                    last_layer = [nn.Linear(input_dim, 2), nn.Sigmoid(),
                                   GumbelMaxBinary(0.1)]
                                # Binary variables
                    if key in self.monotonic_index.keys() and key in self.binary_keys:
                        # TODO: linear layer to transoform the latent part and keey the observed part unchanged. 
                        last_layer = [monotonic_binary_output(len(latent) * latent_out_dim)]  # Binary variables
                self.model_dict[key] = nn.Sequential(*last_layer)
            else:
                if self.binary_keys is None or key not in self.binary_keys:
                    last_layer = [nn.Linear(n_hidden, self.var_dims[key])]

                else:
                    last_layer = [nn.Linear(n_hidden, 2),
                                 nn.Sigmoid(),
                                 GumbelMaxBinary(0.1)]  # Binary variables
                if key in self.monotonic_index.keys() and key in self.binary_keys:
                    # m_ind = []
                    # ind = len(latent) * latent_out_dim
                    # for i in observed:
                    #     if i in self.monotonic_index[key]:
                    #         m_ind.append(list(range(ind,ind + self.var_dims[i])))
                    #     ind += self.var_dims[i]
                    # self.model_dict[key] = min_max_net(input_dim , n_hidden , n_layers, m_ind) # 2 3
                    last_layer = [nn.Linear(n_hidden, 2),
                                 monotonic_binary_output()]  # Binary variables
                else:
                    self.model_dict[key] = nn.Sequential(nn.Linear(input_dim, n_hidden), nn.LeakyReLU(0, inplace=True),
                                                     *block(n_hidden, n_layers - 1),
                                                     *last_layer, )
        self.latent_gen_models = nn.ModuleList([model for key, model in self.latent_gen_dict.items()])
        self.models = nn.ModuleList([model for key, model in self.model_dict.items()])

    
    def _helper_forward(self, z: torch.Tensor, data: torch.Tensor, x: torch.Tensor = None, do_key: int = None):
        """Handles sampling from both observational and interventional data.
        Args:
            z (torch.Tensor): The latent (noise) samples.
            data (torch.Tensor): The empirical observed data from the true distribution. We will not generate the data
                     for root nodes and use the empirical samples instead.
            x (torch.Tensor): The intervention value.
            do_key (int): The intervened variable.
        """
        var = {}

        # Generate latent variables
        w = torch.zeros(z.shape[0], self.latent_out_dim * self.n_latent, device=z.device)
        for i in range(self.n_latent):
            w[:, i * self.latent_out_dim: (i+1) * self.latent_out_dim] =  self.latent_gen_dict[i](z[:, i * self.latent_dim:(i + 1) * self.latent_dim])

        # Generate observed variables
        for key, value in self.graph.items():
            if (do_key is not None) and (key == do_key):
                if x.shape[0] == self.var_dims[do_key]:  # Single intervention for all samples (noise)
                    var[key] = x.reshape((self.var_dims[do_key], -1)).repeat((1, z.shape[0])).t()
                elif x.shape[0] == z.shape[0]:  # Different intervention for each sample (noise)
                    var[key] = x.reshape(-1, x.shape[1])
                else:
                    raise Exception(f'wrong do-var dim. z: {z.shape}, x: {x.shape}')
            else:
                inputs = np.array(value).astype(int)
                if len(inputs) == 0:
                    start = np.sum(self.var_dims[: key])
                    end = np.sum(self.var_dims[: (key + 1)])
                    var[key] = data[:, start:end]   # Empirical data for root nodes
                else:
                    latent = tuple(w[:, (i - 1) * self.latent_out_dim:i * self.latent_out_dim] for i in abs(inputs[inputs < 0]))
                    observed = tuple(var[i] for i in inputs[inputs >= 0])
                    var[key] = self.model_dict[key](torch.cat(latent + observed, dim=1))
                    if self.lower is not None and self.upper is not None and key in self.lower:
                        with torch.no_grad():   # clip the values to lower/upper bound for more stable results
                            lower, upper = self.lower[key].type_as(var[key]), self.upper[key].type_as(var[key])
                            var[key].copy_(var[key].data.clamp(min=lower, max=upper))
        observed = tuple(var[i] for i in range(len(self.var_dims)))
        return torch.cat(observed, dim=1)

    def forward(self, z, data):
        return self._helper_forward(z, data)

    def do(self, z, x, do_key, data):
        return self._helper_forward(z, data, x=x, do_key=do_key)


class MetricsCallback(Callback):
    """Callback to log metrics"""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        metric = dict([(k, each_me[k].cpu().numpy()) for k in each_me])
        self.metrics.append(metric)


class StartEstimandOpt(EarlyStopping):
    """EarlyStopping to finish the pretraining phase and set the value of alpha."""

    def on_validation_end(self, trainer, pl_module) -> None:
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not pl_module.pre_train:
            return
        done, _ = self._run_early_stopping_check(trainer)
        if done:
            pl_module.pre_train = False
            metrics = trainer.callback_metrics
            dist_min = metrics['distance_min_network']
            dist_max = metrics['distance_max_network']
            best1 = min(dist_max, dist_min)
            best2 = max(pl_module.best_dist_min, pl_module.best_dist_max)
            if pl_module.alpha == 0.:
                ## set the value of radius
                ## TODO: add the subsampling trick to decide the value of alpha
                if dist_min > dist_max:
                    pl_module.alpha = self.calculate_radius(pl_module.data_module, pl_module, 1)
                else:
                    pl_module.alpha = self.calculate_radius(pl_module.data_module, pl_module, 0)
                # pl_module.alpha = min(best1, best2) * pl_module.tol_coeff
                print(f"The radius is set to be {pl_module.alpha}")

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> tuple:
        """
        Checks whether the early stopping condition is met
        and if so tells the trainer to stop the training.
        """
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return False, None

        current = logs.get(self.monitor)

        should_stop, reason = self._evaluate_stopping_criteria(current)

        if self.verbose and should_stop:
            self._log_info(trainer, "Estimand optimization starts", False)

        return should_stop, current
    
    def calculate_radius(self, data_module, pl_module: "pl.LightningModule", closest_model: int):
        """Calculate the radius of the ball using subsampling technique"""
        sample_size = 15 * floor(sqrt(data_module.X.shape[0]))
        sample_times = 50
        alpha_list = np.zeros(sample_times)
        for i in range(sample_times):
            ## subsmapling
            idx = np.random.choice(data_module.X.shape[0], sample_size, replace=False)
            sub_data = data_module.X[idx]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU
            sub_data = sub_data.to(device)
            
            z = pl_module._sample_noise(Sample_size = sample_size, imgs = sub_data)
            if closest_model == 0:
                fake = pl_module.generator_min(z, sub_data)
            else:
                fake = pl_module.generator_max(z, sub_data)
            with torch.no_grad():
                r = pl_module._calculate_loss(fake, sub_data).item()
                alpha_list[i] = r
        ## return the 95% quantile of alpha 
        return np.quantile(alpha_list, 0.95)


class LitProgressBar(TQDMProgressBar):
    """A simple progress bar for Lightning."""
    def init_validation_tqdm(self):
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
        import sys
        bar = Tqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=True,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
