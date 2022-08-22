# CapNet
Contains machine learning model for capsule fracture in concrete

## Requirements
- Python (Python 3.9.7 or newer)
- PyTorch (1.11.0 or newer)

## Usage
Please see `Example.py` which has a full toy example, the key details of which are presented below:

### Input Data
The input data consits of five inputs, and should be formatted as:
| Interface Stiffness      | Interface Strength | Interfact Fracture Energy | Capsule Stiffness | Capsule Strength |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 1.67E-01   | 1.00E+00   | 2.50E-01 | 3.33E-03 | 1.67E-01 |
*NB: All inputs are given in relative terms to the matrix data.*

### Network
The network consitis of input, output, and two hidden layers. Five inputs are expected, for interfacial and capsule parameters, and two outputs are given for the case of fracture and non-fracture of the capsule, being 1 and 0 respectively.

### Loading
```{python}
net = Net()
net = torch.load("NNetModel")
net.eval()
```

### Run data through model
```{python}
out = net(data)
outFrac = net(dataFrac)
```

### Print output
```{python}
print(torch.argmax(out).numpy())
print(torch.argmax(outFrac).numpy())
```