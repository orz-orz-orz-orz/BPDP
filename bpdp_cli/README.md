# bpdp_cli: The command line interface of bpdp algrithm
This package is separated from the bpdp package because it needs additional libraries to read the wav/mp3/... files. 

## Installation 
One can install the scripts using pip
```Bash
pip install bpdp_cli
```

## Usage:
1. Extract from single .wav file using default arguments
```Bash
bpdp -i test.wav
```
This will create a test.bpdp.txt contains the epochs.

2. Extract all .wav files from directory
```Bash
bpdp -I test_dir/
```
This will extract all .wav files in the directory recursively. 
For example, if a file test_dir/a/01/a01.wav is in the test_dir/, test_dir/a/01/a01.bpdp.txt will be created.

3. Dry run mode
```
bpdp -I test_dir/ --dry_run
```
This will prevent the actual calculation, file reads and writes.
