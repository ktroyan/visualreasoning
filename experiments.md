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
- ResNet (safe baseline)
- Vanilla ViT
- ViT-ARC
- ViT+registers
- Looped-Transformer (i.e., without ACT)
- Universal-Transformer
- MDLM (Masked Discrete Language Model)

### Experiment Settings
An experiment setting for a (data environment, study) pair is a concrete configuration that determines the dataset, training procedure, evaluation procedure and metrics, as well as the different varying parameters - thus yielding different experiment runs - and additional constraints in order to assess a model's performance in a clear, structured and controlled manner.
<br>

The experiment settings are described in each respective Markdown file (i.e., CVR.md, RE-ARC.md, BEFORE-ARC.md) for each (data environment, study) pair.
<br>

Some experiment settings are approximately shared across data environments, some are exclusive to BEFORE-ARC as it is the only data environment that allows them to be considered.
<br>

### Experiments
An experiment/run/training is a single program run performed for a given model, data environment, experiment setting, train set, and test set.


<br>

# Overview of Experiments
Each item (i.e., experiment setting) in the table below yields a number of experiments/runs for each model.

| Modalities | CVR | RE-ARC | BEFORE-ARC |
|------------|-----|--------|------------|
| **Generalization** | ❓ | ✅ | ✅ |
| **Compositionality** | ✅ | ❌ | ✅ |
| **Sample-Efficiency** &nbsp; &nbsp;| ✅ | ✅ | ✅ |
| **ICL** | ❓  | ❓ | ❓ |

---
