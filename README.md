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
| user_id  | data_num | classes_num | entropy | 0 | ... | 9  | 
|  0   | 5000 | 10| 3.318743| 518                   | ...  | 501 | 
|   ...   |  ... |  ...|  ...|  ... | ... |... |
|  9  | 5000 | 10|3. 3.319255| 503 | ... | 490 | 
