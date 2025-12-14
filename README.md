# Heat Connectivity Retrieval

A minimal Python implementation of a novel **connectivity estimation** method based on **stochastic graph heat modelling**.
This approach infers the underlying graph structure directly from multivariate data by modelling how activity or signals propagate across nodes.

The `connectivity` directory contains the main heat connectivity retrieval function and a few helper functions, while the `examples` directory includes a short example   data.

## Installation

Clone the repository and install the required dependencies to run the connectivity estimation function and examples:

```bash
git clone https://github.com/sgoerttler/Heat_Connectivity.git
pip install -r requirements.txt
```
Alternatively, the `retrieve_heat_graph` function can be used independently by copying the `connectivity` folder into your project and installing the two required dependencies `numpy` and `scipy`.

## Usage
Import the module and use the connectivity retrieval function:
```python
import numpy as np
from connectivity.heat_connectivity import retrieve_heat_graph

# Example data: 20 nodes with 1000 time samples
X_data = np.random.rand(20, 1000)  

# Retrieve 2nd-order heat-based connectivity with regularisation
A_2R = retrieve_heat_graph(X_data, estimation_type='2nd_order', regularisation=True)
```



## Reference
If you use this code in your research, please cite the following paper:

```
@inproceedings{goerttler2024stochastic,
  title={Stochastic Graph Heat Modelling for Diffusion-based Connectivity Retrieval},
  author={Goerttler, Stephan and He, Fei and Wu, Min},
  booktitle={2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages={1--4},
  year={2024},
  organization={IEEE}
}
```

# License
```text
MIT License

Copyright (c) 2025 Stephan Goerttler
```