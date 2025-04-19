# Use
## Data Generation
The file ```generate_cvr_data.py``` can be used to generate CVR data. Those data can then be used to create datasets for experiments. The result of a run creates a folder /generated_data_<img_size>x<img_size> in which folders with the name of the tasks for which examples/samples have been generated can be found.

## Datasets Creation
The file ```create_cvr_datasets.ipynb``` can be used to generate the datasets (train, val, test and gen_test if applicable) for the experiments to perform, as specified in the CVR.md file. The result of a run essentially creates the study folders, experiment settings folders, and experiments folders, in which the dataset splits can be found.

## Note
Run the scripts from the CVR folder.

# CVR Disclaimer
CVR is a Compositional Visual Relations bechmark. We use its code for generating data that we use to conduct experiments.
<br>

The original and official repository can be found [here](https://github.com/serre-lab/CVR).
<br>

Associated paper: [A Benchmark for Efficient and Compositional Visual Reasoning](https://arxiv.org/abs/2206.05379)

 
# License
The CVR code used (and modified) is located in the `/external` folder.

CVR is released under the Apache License, Version 2.0.