
This is the official implementation for the paper Data Valuation and Detections in Federated Learning.

Implementation code is still under development. For more details please contact [wenqian@u.nus.edu](wenqian@u.nus.edu)

## Data Example
You can find the processed data in [this link](https://drive.google.com/file/d/11u_XtfH_Ft_8uVT-1VTfuHmR-i0b4fdE/view?usp=sharing)
### MNIST (IID 10 clients) :    
| user_id  | data_num | classes_num | entropy | 0 | ... | 9  | 
| :---------: | :---------: | :---------: | :-----------: | :----: | :----: | :----: |
|  0   | 5000 | 10| 3.318743| 518   | ...  | 501 | 
|  1   | 5000 | 10| 3.319638 |  503   | ...  | 468 | 
|   ...   |  ... |  ...|  ...|  ... | ... |... |
|  9  | 5000 | 10|3.319255| 503 | ... | 490 | 

### IID Partition:
Each client has the same classes and the same number of classes. Its purpose is to test whether a contribution prediction method yields similar contribution results for approximately similar clients.

### Same Distribution & Different Size:
Data with the same distribution but different sizes; clients 1 and 2 each have 5% of the data; clients 3 and 4 each have 7.5%; clients 5 and 6 each have 10%; clients 7 and 8 each have 12.5%; clients 9 and 10 each have 15%.

### Label Noise:
Flip the labels of a certain proportion of samples within each client. Clients 1 and 2 have no noise; clients 3 and 4 have a noise rate of 5%; clients 5 and 6 have a noise rate of 10%; clients 7 and 8 have a noise rate of 15%; clients 9 and 10 have a noise rate of 20%. Clients with lower noise should have a higher contribution.

### Feature Noise:
Add Gaussian noise to the features of samples within each client. Participants 1 and 2 have no noise; participants 3 and 4 have 5% noise; participants 5 and 6 have 10% noise; participants 7 and 8 have 15% noise; participants 9 and 10 have 20% noise. The lower the noise percentage, the higher the contribution.

### Different Distribution & Same Size:
Different distributions with the same data size, using Dirichlet(1.0) to partition. 
