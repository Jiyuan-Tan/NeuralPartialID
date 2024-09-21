# Partial Identification
This is the code for the experiments of Partial Identification of *the consistency of neural causal partial identification* ([arxiv](https://arxiv.org/abs/2405.15673#:~:text=Recent%20progress%20in%20Neural%20Causal,causal%20graph%20%5BXia%20et%20al.)). The code is adapted from [this code](https://github.com/rgklab/partial_identification).

## How to run
The main programs are in the experiments folder. In the paper, we mainly use the ate_experiment.py. The data generation process is defined in the data folder. If one wants to define his own data generation process, the users can first define the causal graph of the SCM in load_dag.py, then add a function in load_scm.py to specify the data generation process. 
