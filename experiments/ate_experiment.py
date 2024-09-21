import os
import sys
from numba.core.errors import NumbaDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from econml.dml import DML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


import torch
import math
from geomloss import SamplesLoss
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../model'))
sys.path.append(os.path.join(dir_path, '../data'))

# Self-defined Module
from utils import ToyDataModule, get_results, save_results
from common import MetricsCallback, StartEstimandOpt, LitProgressBar
from estimands import create_uatd_gauss, create_ate, create_uniform_ate
from load_scm import gen_scm
from sinkhorn_gn import SinkhornGN


def ATE(setting: str, d0: float, d1: float, n_samples: int, max_epochs: int, seed: int, regular_param:float, radius:float = -1, gpus: int = 1, num_workers: int = 4,
        n_hidden: int = 64, n_layers: int = 3, n_hidden_latent: int = 64, n_layers_latent: int = 3, lr: float = 0.001, lagrange_lr: float = 0.5, ratio: float = 0):
    """Run the Average Treatment Effect experiment.
    Args:
        setting (str): Data-generating process. See ``data.load_scm.gen_scm`` function for options.
        d0 (float): Interval (d0, d1) to calculate the ATE, i.e., ATE = E[Y|do(T=d1)] - E[Y|do(T=d0)].
        d1 (float): Interval (d0, d1) to calculate the ATE, i.e., ATE = E[Y|do(T=d1)] - E[Y|do(T=d0)].
    """
    print(f'Running the ATE experiment with setting {setting} and seed {seed}')
    seed_everything(seed, workers=True)
    batch_size = min(2048, n_samples) # 2048

    interval = (d0, d1)

    # SCM and data
    scm = gen_scm(setting)
    if setting == 'nonlinear_padh_iv_modified':
        data = scm.generate(n_samples, ratio)
    else:
        data = scm.generate(n_samples)
        
    
    tol = 1.2
    # Use g-formula to calculate ATE
    #(l_dml, u_dml) = ATE_g_dml(data[:, 1], data[:, 0], data[:, 2], 0, 1)

    dm = ToyDataModule(torch.tensor(data).float(), batch_size, num_workers)
    #
    # Estimand
    #estimand = create_uatd_gauss(scm.dag, interval=interval, delta=0.1, std=0.5)  # choose std wisely
    estimand =  create_ate(scm.dag, interval=interval) #  create_uniform_ate(scm.dag, interval= interval, width=1e-1)  # create_ate(scm.dag, interval=interval)

    # Model
    monitor = create_ate(scm.dag, interval)
    loss = SamplesLoss(loss="sinkhorn", p = 1, blur=0.01, scaling=0.9, backend='tensorized', diameter=scm.diam)
    model = SinkhornGN(data_module = dm, estimand=estimand, dag=scm.dag, loss=loss, n_hidden=n_hidden, n_layers=n_layers, n_hidden_latent = n_hidden_latent, n_layers_latent = n_layers_latent,
                       lr=lr, lagrange_lr=lagrange_lr, monitor_estimand=monitor, noise = "uniform", tol_coeff=tol, radius=radius, regular_param=regular_param)

    # Trainer
    metrics_callback = MetricsCallback()
    lagrange_start = StartEstimandOpt(monitor='distance_min_network', min_delta=0.004,
                                      patience=max_epochs // 10, verbose=True, mode='min')

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/ATD', every_n_epochs=max_epochs//5)
    prog_bar = LitProgressBar(refresh_rate=20)
    callbacks = [metrics_callback, prog_bar, lagrange_start, checkpoint_callback]
    trainer = Trainer(accelerator='gpu',devices=gpus, auto_select_gpus=True, max_epochs=max_epochs, log_every_n_steps=50, callbacks=callbacks, gradient_clip_val=0)

    trainer.fit(model, dm)

    # Plot function of x
    # plot_final_neural_network_x(model, 0, 1)

    try:
        alpha = model.alpha.to('cpu').item()
    except AttributeError:
        alpha = model.alpha
    distances, estimands = get_results(metrics_callback.metrics, str(monitor),
                                       alpha = alpha, coeff = 1)

    result = {'name': ['ATE'], 'setting': [setting], 'lower_estimand': [estimands[0]], 'upper_estimand': [estimands[1]], 
              #'DR_lower_b': l_dml, 'DR_upper_b': u_dml,
              'seed': [seed], #'min_weight': [weight[0]], 'max_weight': [weight[1]],
              'min_distance': [distances[0]],
              'max_distance': [distances[1]],
              'regular_param': [regular_param],
              'Sample Size': [n_samples],
              'Epoch': [max_epochs],
              'True_ATE': [scm.estimands['ATE']],
              'alpha': [alpha],
              'ratio': ratio,
              'radius': radius}
    save_results(result, 'results/Raw_ATE_Cloud_20240801_leaky' +f'{n_hidden}_{n_layers}_{n_hidden_latent}_{n_layers_latent}_1' +  '.csv')
    return result

def plot_final_neural_network_x(sinkhorn_gn, interval_l:float, interval_h:float):
    plt.figure()
    z = torch.linspace(interval_l, interval_h, 1000).unsqueeze(-1).to(sinkhorn_gn.device)  # 1000 points between -1 and 1
    output = sinkhorn_gn.generator_min.model_dict[0](z).detach().cpu().numpy()
    plt.plot(z.cpu().numpy(), output)
    plt.savefig('results/backdoor_x_function_unreg_20240219.png') 

def ATE_g_dml(T: ndarray, X: ndarray, Y: ndarray, d0: float, d1: float):
    """ Regression for the Backdoor graph. Assume that X is constant.  """

    est = DML(
        model_y='auto',
        model_t='auto',
        model_final=StatsModelsLinearRegression(fit_intercept=False),
        linear_first_stages=False,
        discrete_treatment= True
    )
    est.fit(Y.reshape(-1, 1), T.reshape(-1, 1), X=X.reshape(-1, 1), W=None)
    return est.ate_interval(X = X.reshape(-1, 1))

def Test_plot (setting: str, regular_param:list, Sample_size:list,seed: list,plt_baseline:int, plt_color: str, max_epochs:int =500, radius: float =-1, ratio: float = 0):
    upper_est, lower_est = np.zeros((len(seed), len(Sample_size))), np.zeros((len(seed), len(Sample_size)))
    upper_est_dr, lower_est_dr = np.zeros((len(seed), len(Sample_size))), np.zeros((len(seed), len(Sample_size)))

    for i in range(len(seed)):
        for j in range(len(Sample_size)):
            result = ATE(setting, 0, 1, Sample_size[j], max_epochs, seed[i],regular_param =  regular_param[j], num_workers = 4, gpus = 1, radius = radius, n_hidden = 128, n_layers = 3, n_hidden_latent = 128, n_layers_latent = 6, ratio = ratio)
            # 128 3
            upper_est[i, j] = result['upper_estimand'][0]
            lower_est[i, j] = result['lower_estimand'][0]
            #lower_est_dr[i, j] = result['DR_lower_b']
            #upper_est_dr[i, j] = result['DR_upper_b']

    upper_err = np.mean(upper_est, axis=0)
    lower_err = np.mean(lower_est, axis=0)
    #upper_err_dr = np.mean(upper_est_dr, axis=0)
    #lower_err_dr = np.mean(lower_est_dr, axis=0)
    # ate_g = np.mean(ate_g, axis=0)

    x = Sample_size

    result_csv = {'setting': [setting]*len(Sample_size), 'lower_estimand': lower_err, 'upper_estimand': upper_err, 
                  #'lower_estimand_dr': lower_err_dr, 'upper_estimand_dr': upper_err_dr, 
                  'True_ATE': np.full(len(Sample_size), result['True_ATE']), 'regular_param': regular_param, 'Sample size': Sample_size, 'Epoch': np.full(len(Sample_size), max_epochs), 'radius': radius}
    save_results(result_csv, 'results/ATE_l&u_Cloud_20240801_leaky.csv')
    
    # plt.figure()
    y_mean_wass = (lower_err + upper_err)/2
    asymmetric_error = [y_mean_wass - lower_err, upper_err - y_mean_wass]
    #plt.errorbar(x, y_mean_wass, yerr=asymmetric_error, fmt='o', ecolor= plt_color, capsize=5, linestyle='-', color=plt_color)

    if plt_baseline == 1:
      y = np.full(len(Sample_size), result['True_ATE'])
      plt.errorbar(x, y, linestyle='-', color='red')

      y_mean_dr = (lower_err_dr + upper_err_dr)/2
      asymmetric_error = [y_mean_dr - lower_err_dr, upper_err_dr - y_mean_dr]
      plt.errorbar(x, y_mean_dr, yerr=asymmetric_error, fmt='o', ecolor='green', capsize=5, linestyle='-', color='green')

      plt.title('Test on One Counteraxmple')
      plt.xlabel('Sample Size')
      plt.ylabel('ATE')
      plt.xticks(x)
    
    
def save_training_data(setting: str, n_sample: int, seeds:int, ratios):
    i = 0
    for seed in seeds:
        for ratio in ratios:
            seed_everything(seed, workers=True)
            scm = gen_scm(setting)
            if setting == 'nonlinear_padh_iv_modified':
                data = scm.generate(n_sample, ratio)
            else:
                data = scm.generate(n_sample)
            result = {'Z':data[:,0],'X':data[:,1].astype(np.int64),'Y':data[:,2]}
            save_results(result, 'results/leaky' +f'_{n_sample}_{i}' +  '.csv')
            i += 1


if __name__ == '__main__':
    # fire.Fire(ATE)
    seeds = [16, 17, 18, 19, 20]
    seed_everything(1, workers=True)
    Sample_size = [5000] #[500, 1000, 2000, 5000]
    ratios = np.random.rand(10)
    regular_param = [0]*len(Sample_size)
    setting = "leaky"    #"backdoor_counterexample"  #"binary_iv_mono"  #"nonlinear_padh_iv_modified" # "binary_iv" #"backdoor_counterexample_binary"
    max_epochs = 600
    
    
    
    #Test_plot(setting, regular_param, Sample_size, seeds, 0, 'blue', max_epochs, radius = 5)
    #Test_plot(setting, regular_param, Sample_size, seeds, 0, 'blue', max_epochs, radius = -1)
    #Test_plot(setting, regular_param, Sample_size, seeds, 0, 'k', max_epochs, radius = 8)
    Test_plot(setting, regular_param, Sample_size, seeds, 0, 'blue', max_epochs, radius = 8)
    
    # save_training_data(setting, 5000, seeds, [0])
    
    #for ratio in ratios:
        #Test_plot(setting, [0]*len(Sample_size), Sample_size, seeds, 0, 'blue', max_epochs, radius = 8, ratio = ratio)
        # Test_plot(setting, regular_param, Sample_size, seeds, 0, 'k', max_epochs, radius = 6)
