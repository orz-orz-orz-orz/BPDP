# BPDP: Epoch extraction with Band Pass filter and Dynamic Programming

This is the official python implementation of our algorithm. 

This repo contains two packages:
- bpdp: the python library.
- bpdp_cli: the command line interfaces.

## Installation
You can install the libary and CLI by:
```
pip install bpdp bpdp_cli
```

If you only need the python library:
```
pip install bpdp
```

## Documentation and Usage
For bpdp see [bpdp/README.md](bpdp/README.md)

For bpdp_cli see [bpdp_cli/README.md](bpdp_cli/README.md)


## Experiments
Most of the codes of our experiments are put in the notebooks/ directory. 

## Other Data
We also provides 30 manually checked and modified TextGrid files of CMU ARCTIC used in the article.
The voiced/unvoiced annotations are fully marked by hand. The reference GCIs are extracted from the EGG signal using BPDP, then corrected manually. 
The GCIs in the unvoiced part are keep unchecked and should not be used.
All Files are put in data/CMU_ARCTICS_SJB30


