# Installation 
## Requirements
- In Ubuntu16.04, install **CUDA 10.0.13 and CUDNN 7.4.2**.
- Create a virtual environment.

```
conda create -n PVENcoNet python=3.6
conda activate PVENcoNet
```

- Install pytorch.

```
conda install pytorch==1.3.0
```

- Clone this repository.

```
git clone https://github.com/XYunaaa/PV_ENcoNet.git
cd PV_ENcoNet
```


- Install the dependent python libraries.

```
pip install -r requirements.txt
cd nearest_neighbors
python setup.py install --home="."
cd ..
```

- Install the SparseConv library, we use the implementation from [spconv](https://github.com/traveller59/spconv).
    - If you use PyTorch 1.3+, then you need to install the spconv v1.2.
    As mentioned by the author of [spconv](https://github.com/traveller59/spconv), you need to use their docker if you use PyTorch 1.4+.

- Our code is based on pcdet's code. Install this pcdet library by running the following command.

```
cd PV_ENcoNet's path
python setup.py develop
```

