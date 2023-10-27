Implementations for 'An Optimal Transport Approach to Personalized Federated Learning'



10.6 Update barycenter calculations for MNIST datasets in folder WBTransport, the reference paper is 'Wasserstein Barycenter for Multi-Source Domain Adaptation'
https://openaccess.thecvf.com/content/CVPR2021/papers/Montesuma_Wasserstein_Barycenter_for_Multi-Source_Domain_Adaptation_CVPR_2021_paper.pdf

10.19 Update barycenter calculations under federated learning in folder FedWad, the reference paper is 'Federated Wasserstein Distance' https://arxiv.org/pdf/2310.01973.pdf

10.25 Update Federated otdd for MNIST datasets in folder FedWad. 
Todo: 
1. implement FedLAVA with available testing data 
2. implement FedBary without testing data
3. Domain Adaptation applications 

MNIST iid setting : 
   user_id  data_num  classes_num   entropy    0  ...    5    6    7    8    9
0        0      5000           10  3.318743  518  ...  459  516  515  475  501
1        1      5000           10  3.319638  503  ...  465  486  516  470  468
2        2      5000           10  3.317587  519  ...  410  508  527  504  490
3        3      5000           10  3.318782  432  ...  489  502  501  517  499
4        4      5000           10  3.315813  540  ...  408  513  518  494  481
5        5      5000           10  3.316716  476  ...  423  495  523  469  486
6        6      5000           10  3.316706  472  ...  418  519  480  472  519
7        7      5000           10  3.319383  496  ...  493  436  539  478  536
8        8      5000           10  3.319665  473  ...  468  462  522  484  518
9        9      5000           10  3.319255  503  ...  473  514  534  479  490
