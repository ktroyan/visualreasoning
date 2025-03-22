# Use
## Data Generation
The file ```generate_rearc_data.py``` can be used to generate REARC data. Those data can then be used to create datasets for experiments. The result of a run creates a folder /generated_data_lb<lb_value>_ub<ub_value> in which a metadata.json and a /tasks folder in which tasks <task_name>.json files contain the generated samples/examples.

Note that there should be a folder ```external/arc_original``` containing the original training and evaluation ARC tasks. Hence, the file ```arc_original.zip``` is included and unzipping it yields the aforementioned folder. Examples of how the original ARC tasks look can be visualized [here]([here](https://kts.github.io/arc-viewer/)).

## Datasets Creation
The file ```create_rearc_datasets.ipynb``` can be used to generate the datasets (train, val, test and test_gen if applicable) for the experiments to perform, as specified in the REARC.md file. The result of a run essentially creates the study folders, experiment settings folders, and experiments folders, in which the dataset splits can be found.

## Note
Run the scripts from the REARC folder.


# RE-ARC Disclaimer
RE-ARC is Reverse-Engineering the Abstraction and Reasoning Corpus. We use its code for generating data that we use to conduct experiments.
<br>

The original and official repository can be found [here](https://github.com/serre-lab/CVR).
<br>

Associated paper: [Addressing ARC via Procedural Example Generation](https://arxiv.org/abs/2404.07353)

 
# License

RE-ARC is released under the MIT License.
