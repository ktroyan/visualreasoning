# visualreasoning
Evaluating Neural Network architectures by studying Systematic Generalization, Sample Efficiency, Compositionality and ICL in Abstract Visual Reasoning (AVR) tasks from different data environments.

## Repository Structure Overview
```
├── .python-version
├── BEFOREARC
├── CVR
│   ├── CVR.md
│   ├── create_cvr_datasets.ipynb
│   ├── external
│   └── generate_cvr_data.py
├── REARC
│   ├── REARC.md
│   ├── create_rearc_datasets.ipynb
│   ├── external
│   ├── generate_rearc_data.py
│   └── tasks_generated.txt
├── configs
│   ├── base.yaml
│   ├── data.yaml
│   ├── experiment.yaml
│   ├── inference.yaml
│   ├── models
│   │   ├── CVRModel.yaml
│   │   └── REARCModel.yaml
│   ├── networks
│   │   ├── backbones
│   │   └── heads
│   ├── sweep_rearc.yaml
│   ├── sweep_cvr.yaml
│   ├── training.yaml
│   └── wandb.yaml
├── data
│   ├── cvr_data.py
│   ├── data_base.py
│   └── rearc_data.py
├── experiment.py
├── figs
├── inference.py
├── jobs
├── models
│   ├── cvr_model.py
│   └── rearc_model.py
├── networks
│   ├── backbones
│   └── heads
├── pyproject.toml
├── requirements.txt
├── training.py
├── utility
│   ├── cvr
│   ├── logging.py
│   ├── rearc
│   └── utils.py
└── uv.lock
```

## Environment Setup
### 1. Clone Repository  
Clone the ```visualreasoning``` repository to your machine:
```sh
git clone https://github.com/ktroyan/visualreasoning.git
```
Enter the ```visualreasoning``` repository folder:
```sh
cd visualreasoning
```

### 2. Create Virtual Environment
Using [```uv```](https://docs.astral.sh/uv/), run the following commands to install the necessary dependencies (Python version, libraries).

Create a virtual environment:
```sh
uv venv
```

Activate the virtual environment ```.venv```:
```sh
source .venv/bin/activate   # Linux

.venv\Scripts\activate      # Windows
```

### 3. Install Dependencies  
Install/update the dependencies from the pyproject.toml file:
```sh
uv sync
```

That's it!

## Usage
### Preliminary
The working directory is assumed to be ```/visualreasoning```, unless changing directory is explicitly mentioned.

---

**TL;DR**  
If you do not have access to the correctly generated data, run the following command to generate them:
```sh
# For REARC
cd REARC
uv run generate_rearc_data.py

# For CVR
cd CVR
uv run generate_rearc_data.py
```

Then, create the experiment datasets using the notebook:
- In the REARC directory, run the complete notebook ```create_rearc_datasets.ipynb```
- In the CVR directory, run the complete notebook ```create_cvr_datasets.ipynb```

This will create experiment datasets at the leaf folders of the folder ```/final_datasets``` created in the respective data environment folder.

The model experiments can now be run using those datasets. See the section [Scripts](#scripts).

---
#### Data
To be able to execute the different programs, correctly prepared data must be available. Such data are contained in a folder with dataset splits (of either JSON or CSV format depending on the data environment considered).

More specifically, experiment folders containing data splits (i.e., train, val, test, and optionally gen_test) must exist in some folder of path ```./<data_env>/final_datasets/<study>/exp_setting_<index>/<experiment_name>```. 

For example:  
```./REARC/final_datasets/sample-efficiency/exp_setting_1/experiment_2```

To create the datasets for a given data environment, the notebook ```create_<data_env>_datasets.ipynb``` can be executed, where <data_env> is either "cvr" or "rearc". Make sure that data were generated beforehand using the ```generate_<data_env>_data.py``` script.

There is also the possibility to directly generate the data in the same notebook run. For this, in the notebook there is a global variable flag that decides wether data must be generated (using the data environment generator) or not. If the flag is set to False, it assumes that generated data already exist. If the flag is set to True, the notebook will first generate data given the parameters in the notebook before creating the datasets.

#### Scripts
There are essentialy three scripts that can be run:  
- ```experiment.py``` to perform a full run of a (tracked) experiment with training, testing, and logs
- ```training.py``` to only train a model
- ```inference.py``` to only test a model

See section [Program](#program) below for more information about running the ```experiment.py``` script.

#### Parameters
All the config parameters for the different modules are found in the relevant YAML config files in ```/configs```.


## Program
### Simple Run
An experiment is run as:
```bash
uv run experiment.py
```

[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) CLI arguments can be used to overwrite the config arguments set in the config files located in ```/configs```.
For example:
```bash
uv run experiment.py base.data_env="REARC" experiment.study="sample-efficiency" experiment.setting="exp_setting_1" experiment.name="experiment_1" training.max_epochs=5
```
This performs an experiment run for the REARC data environment for a Sample-Efficiency study of experiment setting and of some experiment part of that experiment setting. The model is trained for 5 epochs (at most).

The study, experiment setting and experiment are defined explicitly in the file `create_<data_env>_datasets.ipynb` and the associated data environment README. Moreover, some useful metadata can be found in the created experiment folders.

The result of an experiment run is complete logs of the configs, data, metrics, plots, etc.


### Sweeps
There is the possibility to perform experiments sweeps using [WandB Sweeps](https://docs.wandb.ai/guides/sweeps/). For this, simply set the parameters ```wandb.sweep.enabled ``` to true and the parameter ```wandb.sweep.num_jobs``` to the number of parameter combinations to run experiments for. The sweep config can be found in ```/configs/sweep_<data_env>.yaml```.


# General information
The goal is to study Transformer-based vision models for Abstract Visual Reasoning (AVR) tasks across different data environments and experiment settings. 
<br>

The experiment settings are defined with respect to combinations of the data environments and studies. Each experiment setting yields a concrete number of experiments (i.e., runs) for each model.
<br>

Each model should be evaluated (i.e., trained and tested) for all experiment settings and the models are compared for each experiment setting.

# Terminologies
### Data environments
A data environment is a source of AVR tasks with its own definitions, properties and constraints. Essentially, it contains images which are related by some defined AVR rules/transformations.
<br>

The data environments considered are the following: 
- CVR
- RE-ARC
- BEFORE-ARC  

### Studies
A study is a specific aspect of research that is highly relevant to the problem/topic considered and aligns with the overall research goals. Hence, here, a study refers to a focused investigation of key properties that help assess the capabilities and performance of a model given a data environment.
<br>

The studies considered are the following: 
- Systematic Generalization (difficulty-split)
- Compositionality
- Sample-Efficiency
- In-Context Learning (ICL)

Note that we sometimes write "Generalization" to shorten "Systematic Generalization".

### Models
A model is considered in order to evaluate its capabilities at learning AVR tasks and share its performance.
<br>

The models considered are the following: 
- [ResNet](https://arxiv.org/pdf/1512.03385)
- [Vanilla ViT](https://arxiv.org/pdf/2010.11929)
- [ViT+registers](https://arxiv.org/pdf/2309.16588)
- [ViTARC](https://openreview.net/pdf?id=0gOQeSHNX1)
- Looped-Transformer (i.e., U-T without ACT)
- [Universal-Transformer](https://arxiv.org/pdf/1807.03819)
- [MDLM](https://arxiv.org/pdf/2406.07524)

### Experiment Settings
An experiment setting for a (data environment, study) pair is a concrete configuration that determines the dataset, training procedure, evaluation procedure and metrics, as well as the different varying parameters - thus yielding different experiment runs - and additional constraints in order to assess a model's performance in a clear, structured and controlled manner.
<br>

The experiment settings are described in each respective Markdown file (i.e., CVR.md, RE-ARC.md, BEFORE-ARC.md) for each (data environment, study) pair.
<br>

Some experiment settings are approximately shared across data environments, some are exclusive to BEFORE-ARC as it is the only data environment that allows them to be considered.
<br>

### Experiments
An experiment for a tuple (data environment, experiment setting, train set, and test set) is a specific configuration where the Experiment Setting's parameters of interest are set for that experiment. Hence, the folder $experiment\_i$ in $experiment\_setting\_j$ contains the dataset splits (i.e., train, val, test, and eventually gen_test).

### Runs
A run is a single program run performed for a given model and experiment.


<br>

# Overview of Experiments
Each item (i.e., experiment setting) in the table below yields a number of experiments/runs for each model.

| Modalities | CVR | RE-ARC | BEFORE-ARC |
|------------|-----|--------|------------|
| **Generalization** | ✅* | ✅ | ✅ |
| **Compositionality** | ✅ | ❌ | ✅ |
| **Sample-Efficiency** &nbsp; &nbsp;| ✅ | ✅ | ✅ |
| **ICL** | ❓  | ❓ | ❓ |

<br>

\* We use the generalization test set defaulting from CVR. However, we argue that it is not a good generalization split as the parameters cannot be controlled *in a simple and straightforward way* (e.g., we cannot set the number of objects), making the increased difficulty of the samples in the generalization test set hard to grasp and use meaningfully.

# Summary of Experiments

| | Compositionality | Sample-Efficiency | Generalization |
|-------------------|-----------------|------------------|---------------|
| **RE-ARC**       | ❌  | **Problem:** <br> Image (grid) Generation with Learn/Apply <br> <br> **Setting 1*** with 10 different tasks sampled randomly  | **Problem:** <br> Image (grid) Generation with Learn/Apply <br> <br> **Setting 1** with 10 different tasks sampled randomly  |
| **BEFORE-ARC**   | **Problems:** <br> - Image (grid) Generation with Task Index Vector <br> - Image (grid) Generation with fixed number of Demonstrations <br> <br> Using Demonstrations Examples or using Task Index Vector to let the model know what transformation to consider. <br> <br> **Tasks:** <br>Translate-Up, Rot-90, Mirror-Hor, Crop-Top and their depth 1 composition <br> <br> **Setting 1** with tasks in Tasks <br> <br> **Setting 2** with tasks in Tasks <br><br> **Setting 3**** with tasks in Tasks <br><br> **Setting 5** with tasks in Tasks as well as the depth $2$ compositions of the primitive tasks in Tasks | **Problem:** <br> Image (grid) Generation with Learn/Apply <br> <br> **Tasks:** <br> Translate-Up, Rot-90 (and, if time allows, Mirror-Hor and Crop-Top) <br> <br> **Setting 1*** with tasks in Tasks  | **Problem:** <br> Image (grid) Generation with Learn/Apply <br> <br> **Tasks:** <br> Translate-Up, Rot-90 (and, if time allows, Mirror-Hor and Crop-Top) <br> <br> **Setting 1** with tasks in Tasks <br> <br> **Setting 2** with tasks in Tasks |
| **CVR**         | **Problem:** <br> Classification with Task Index Vector <br> Using Task Index Vector to let the model know what rule to consider. <br><br> **Tasks:** <br> Pos, Rot, Count, Color and their depth 1 combination (with lower index in the composite task name) <br><br> **Setting 1** with tasks in Tasks <br> <br> **Setting 2** with tasks in Tasks <br><br> **Setting 3**** with tasks in Tasks | **Problem:** <br> Classification with Learn/Apply <br><br> **Tasks:** <br> Pos, Rot (and, if time allows, Count and Color) <br><br> **Setting 1*** with tasks in Tasks <br> |   **Problem:** <br> Classification with Learn/Apply <br><br> **Tasks:** <br> Pos, Rot (and, if time allows, Count and Color) <br><br> **Setting 1** with tasks in Tasks |

### Notes

\* But we should also save the results for Setting 2.

\*\* Add a flag in the code so that Setting 4 (for which the only difference with Setting 3 is that it only considers the elementary transformations/tasks part of the composite transformations/tasks on which the model is trained but not the elementary transformations/tasks part of the composite transformation/task that the model has to predict).

The table above refers to the Markdown files (CVR.md, RE-ARC.md, BEFORE-ARC.md) uploaded on this repo ("visualreasoning").

Regarding the Problems for BEFORE-ARC that involve image generation, we could consider an MCQ mode or a "predict the transformation(s)" mode. This would be easier for the model than having demonstration examples as context input and would also show what can be done with BEFORE-ARC.
