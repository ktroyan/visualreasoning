# CVR Experiment Settings
The experiment settings are described in each study section below for the CVR data environment.
<br>

There are several problems that could be considered, however the CVR code only allows a straightforward generation of data suited for the problem of "odd-one-out". We note that CVR theoretically also allows for the creation of problems such as binary classification, few-shot binary classification and Raven's progressive matrix, where the last one would be relevant with respect to the other data environments (i.e., RE-ARC and BEFORE-ARC) considered. <br>

The problem considered is: 
- Categorical prediction given a set of images: Odd-one-out
  - A sample contains $k=4$ RGB input images following an AVR rule, and the output is a categorical value indicating which of the $k$ images is the odd-one (i.e., the one not following the same AVR rule as the others).

<br>

Note that $k$ can be changed in the code in the file tasks.py (or generalization_tasks.py) through the variable $n\_ samples$.

A rule is associated to a task. There are elementary rules and composite ones. The composite rules are either roughly the result of a combination (e.g., task_pos_contact or task_size_count_2), or they are a "standalone" more complex rule (e.g., task_sym_mir). If a task name contains an index at the end of its name, it is an additional rule variant of a "rough" composition of elementary rules.
<br>

Note that we changed the task names of two tasks due to what seems to be inconsistencies: task 53 "task_pos_col_1" to "task_pos_color_1" and task 54 "task_pos_col_2" to "task_pos_color_2".
<br>

Constraints of the CVR data environment are:
- ...
- ...
<br>

<br>

The considered varying parameters are:
- number of images per sample
- images size

→ However we do not vary any of those parameters and fix them to $4$ for the number of images per sample and $(128,128)$ for the image size. The fixed values are as per the CVR default.


## Systematic Generalization
Consider $T_{elem}=2$ of the $9$ elementary tasks.
<br>

The elementary rules/tasks considered are: Pos and Rot. If time allows, we will consider more elementary tasks in addtion (e.g., Count and Color).
<br>

We justify the choice of elementary tasks by stating that the increased complexity of considering composite tasks could confound the study of the ability of the model to generalize OOD in case the failure would stem from the complexity of the composite samples instead of the increased difficulty of the systematic generalization test set. Moreover, we only consider $2$ of the elementary tasks due to limited resources.
<br>

CVR does not allow a straightforward controlled modification of the level of difficulty/complexity of the generated samples. However, it it possible to generate data said to be of higher difficulty by using their default code (see generalization_tasks.py). The authors of CVR state that "A higher number of random parameters results in a higher difficulty.". Thus, they allow to create a Systematic Generalization test set by using the generalization_tasks.py file where the sets of fixed and random parameters are changed compared to that in tasks.py used for standard in-domain testing (i.e., where the samples seen during training are of same difficulty as those seen at test time).
<br>

Consequently, there is no direct or simple way to generate samples for the Systematic Generalization study for a difficulty-split based on the number of objects (→ there is no explicit and consistent parameter for that across all the tasks) or the grid size (→ image dimension in this case, which can in fact easily be defined but does not seem relevant due to how Vision Transformers would process such images with patches leading to downsampling and thus a somehwat useless change of image size).

<br>

**Experiment setting 1: Various Parameters Randomness Difficulty**<br>
For each task considered:
- Train on $N_{train}$ samples with the standard train set (resulting from tasks.py)
- Test on $N_{test}$ samples with the systematic generalization test set (resulting from generalization_tasks.py)

<br>


## Compositionality
Consider $T_{elem} = 4$ of the $9$ elementary tasks and a related composite pair for each combination of depth $1$. This yields $T_{composite} = \sum_{i = 1}^{T_{elem}} i = \frac{T_{elem}^2 + T_{elem}}{2} = 10$ composite tasks selected, since the rules' order does not matter, we consider all unique pairs between the tasks, and repetitions are allowed.
<br>

The elementary rules/tasks considered are: Pos, Rot, Count and Color. If time allows, we will consider more tasks in addition.
<br>

Note that we do not write "the" related composite pair as there can apparently be more than one &ndash;  the composite task name would contain an index at its end and not all elementary pairs have an associated "pure" composition, hence we select the composite task with the lower index (or none).

<br>

**Experiment setting 1: Composition from Elementary to Composite**<br>
For each pair of elementary tasks in the set of tasks that are elementary tasks making up their associated composite task:
- Train on $N_{train}$ samples from the relevant elementary tasks
- Test on $N_{test}$ samples from the associated composite task

<br>

**Experiment setting 2: Decomposition from Composite to Elementary**<br>
For each composite task in the set of tasks that are composite tasks associated with the composition of two elementary tasks:
- Train on $N_{train}$ samples from the composite task
- Test on $N_{test}$ samples from the associated elementary tasks

<br>


**Experiment setting 3: From Composite to Unseen Composite**<br>
For each composite task that is the depth $1$ composition of two elementary rules:
- Train on $N_{train}$ samples from all the relevant composite tasks except one
- Test on $N_{test}$ samples from the relevant composite task not seen during training

<br>

**[Not Considered] Experiment setting 4: From Composite and Restricted Elementary to Unseen Composite**<br>
For each elementary task and composite task that is the depth $1$ composition of two elementary rules:
- Train on $N_{train}$ samples from all the relevant tasks except one composite task and the elementary tasks composing it
- Test on $N_{test}$ samples from the relevant composite task not seen during training


## Sample-Efficiency
Consider $T_{elem} = 2$ of the $9$ elementary tasks.
<br>

The elementary rules/tasks considered are: Pos and Rot. If time allows, we will consider more elementary tasks in addition (e.g., Count and Color).
<br>

We justify the choice of elementary tasks by stating that considering composite tasks would not provide consequent insights on the sample-efficiency ability of the model given that we know from the other experiments part of the Compositionality study that the model can learn to some extent the composite samples from the elementary ones. Moreover, we only consider $2$ of the elementary tasks due to limited resources.
<br>

We have to decide on one of the following modalities that will decide on a single associated experiment setting below:
- Train separately on datasets of size $N_1, ..., N_k$ (e.g. $N_1=100, N_2=500, ..., N=10000$) different samples respectively and write the best performance for each dataset size. We train until convergence and use early stopping or save the best checkpoint.

- Train on a dataset of sufficiently large $N$ (e.g. $N=10000$) samples and report the model performance when $n_1, n_2=..., n_k$ (e.g. $n_1=100, n_2=500, ..., n_k=10000$) **different** samples have been seen.

- Train on a dataset of fairly large $N$ (e.g. $N=5000$) samples and report the number of steps a model has performed to reach its best performance. 
<br>

→ We decide to consider the first modality for the following experiment settings.

<br>

**Experiment setting 1: Best Performance for Varying Number $N$ of Samples**<br>
For each task considered and each $N \in \{100, 500, 1000, 2500, 5000, 10000\}$:
- Train on $N_{train} = N$ samples until convergence
- Test on $N_{test}$ samples with the best model checkpoint obtained during training

<br>

**[Considered within 1] Experiment setting 2: Best Performance across $N$ Different Samples**<br>
Fix a sufficiently large $N=10000$. Consider $n_i \in \{100, 500, 1000, 2500, 5000, 10000\}$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples
- Test on $N_{test}$ samples after the model has seen $n_i$ **different** samples during training

<br>

**[Not Considered] Experiment setting 3: Best Performance across $S$ Steps**<br>
Fix a fairly large $N=5000$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples until convergence of the model
- Test on $N_{test}$ samples with the best model checkpoint obtained during training and report how many steps had been performed by that model checkpoint



## ICL
See later.