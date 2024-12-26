# lid_driven_flow
template folder for the application of physics and equality constrained artificial neural networks (PECANNs) in lid driven cavity flow

Frequent changes would happens to improve the code and performance.

## Codes can be executed by run.slurm or just main.py.
Default: 3 independent networks representing u,v,p.
         
         Re=100

Uniform mesh Failed.
Recent version is to test the consistency of prediction of 4 hidden layers and 30 neurons per layer with random collocation points.
  
  It works very consistently, for both 4096 and 10000 collocation points.
  
  The total model parameters are 8733.

  Following this, we gonna test Re=400.
