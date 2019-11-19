# PlackettConfidence
The present code computes the parameter vector and metric result under the Plackett Luce model.


The code in this repository provides a framework for variational inference computation ranking data. In particular, the algorithms implemented are described in the paper: 

  > ["Variational Inference from Ranked Samples with Features"](https://ece.northeastern.edu/fac-ece/ioannidis/static/pdf/2019/C_Guo_Variational_ACML_2019.pdf), Yuan Guo, Jennifer Dy, Deniz Erdogmus,
  > Jayashree Kalpathy-Cramer, Susan Ostmo, J.Peter Campbell, Michael F.Chiang and Stratis Ioannidis. In Asian Conference on   
  > Machine Learning, pp. 599-614. 2019. 
  
Please cite this paper if you intend to use this code for your research.





## mathpackage.py ##
The python file includes the following modules:
```
numpy
scipy
random
```
##### Function `EMPlackett`: #####
The EM function to compute the variational inference mean and covariance matrix.

This file will return the mean, covariance matrix and lower bound. The input variables are:

```
(Xarray,RankPlack,C_value,args.loopT) 
```

* `Xarray` is the feature matrix for N absolute samples.

* `RankPlack` is the dictionary for the ranking index.

* `C_value` is a variable for prior Gaussian distribution .

* `args.loopT` is the iteration number for inner altermation.



##### Function `MapEstimation`: #####
The Newton method to compute the parameter estimation of MAP.

This file will return the parameter vector. The input variables are:

```
(Xarray,RankMul,C_value) 
```

* `Xarray` is the feature matrix for N absolute samples.

* `RankMul` is the dictionary for the ranking index （with top query form).

* `C_value` is a variable for prior Gaussian distribution .


## Acknowledgement

Our work is supported by NIH (R01EY019474, P30EY10572), NSF (SCH-1622542 at MGH; SCH-1622536 at Northeastern; SCH-1622679 at OHSU), and by unrestricted departmental funding from Research to Prevent Blindness (OHSU).
