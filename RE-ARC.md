# RE-ARC Experiment Settings
The experiment settings are described in each study section below for the RE-ARC data environment.
<br>

The problem considered is:
- Image (grid) generation
  - A sample contains an input (grid) image and the output is an image obtained by a ground-truth AVR transformation (although the transformation underlies the input-output pair w.r.t. the RE-ARC code) of the input image.

<br>

A transformation is defined algorithmically for each of the $400$ tasks of the ARC. The $400$ transformations are not explicitly classified or structured, for example by having elementary transformations and more complex ones such as composite transformations.
<br>

Constraints of the RE-ARC data environment are:
- The maximum grid size is $(30,30)$
- The maximum number of symbols/colors per grid image is $10$
<br>

<br>

The considered varying parameters are:
- Difficulty interval (specified by a lower bound and upper bound), where the closer to $0$, the less difficult the sample, the closer to $1$, the more difficult the sample.
  - The difficulty interval impacts cardinalities &ndash; used as proxy for the difficulty/complexity of a sample &ndash; of sampled parameters such as grid height, grid width, number of symbols, number of objects, sizes of objects, number of symbols per object, distances "to travel" for an object, etc. However, the aforementioned parameters do not necessarily appear in all tasks.
  - The difficulty intervals considered when creating "levels" of difficulty/complexity should be very few, as otherwise "...the ranges of possible values may be too small and would
render the pruned range empty" and "...in many cases, some degrees of freedom are not independent.". Hence, we can create two or three difficulty levels by using the intervals $interval_{1} = [0, 0.7], interval_{2} = [0.7, 1]$ or $interval_{1} = [0, 1/3], interval_{2} = [1/3, 2/3], interval_{3} = [2/3, 1]$, respectively.

## Systematic Generalization
Sample $T=10$ tasks from the $400$ tasks. This can be done either completely randomly or randomly for a subset of tasks where the tasks are selected according to some criterion. Moreover, we justify selecting $T$ tasks randomly by stating that there is no explicit structure or hierarchy in the $400$ ARC tasks and sampling randomly for $T$ large enough (e.g., $T=10$) should yield results representative of what would be obtained on average on the whole set of $400$ tasks.
<br>

For example, the tasks have something in common with the tasks of the other data environments. <br>

Another example would be to select the tasks for which the difficulty/complexity level (defined through an interval $[lb, ub]$) mainly impact the grid size (this would actually be all the tasks) and the number of objects, as that's what is considered in BEFORE-ARC for the study of Systematic Generalization. We would thus consider tasks in a similar mode as the ones in BEFORE-ARC, hence allowing a possibly more related comparison across data environments.
<br>

Moreover, we note that we could instead use some difficulty metric post data generation in order to create the datasets of different difficulty, but choosing the interval difficulty using cardinality of parameters as a proxy for difficulty makes sense and is straightforward.

<br>

**Experiment setting 1: Various Parameters Cardinality Difficulty**<br>
Define the two intervals $interval_{1} = [0, 0.7], interval_{2} = [0.7, 1]$, which determine the two levels of difficulty/complexity. 
<br>

For each task considered:
- Train on $N_{train}$ samples generated with $interval_{1} = [0, 0.7]$
- Test on $N_{test}$ samples generated with $interval_{2} = [0.7, 1]$

<br>


## Compositionality
None.
<br>

There is no direct way to obtain compositions from the RE-ARC generator, especially since there is no clear/explicit elementary/primitive transformation.

## Sample-Efficiency
Select $T=10$ tasks randomly from the $400$ tasks and generate the samples with a low level of difficulty/complexity of $interval = [0, 0.4]$. The idea is that a sample would represent a task closer to "elementary" tasks &ndash; which is also the choice we made in the CVR and BEFORE-ARC data environments &ndash; in order to avoid confounding the learning performance with the difficulty of the tasks (e.g., if the tasks are "composite" as opposed to "elementary"). Moreover, we justify selecting $T$ tasks randomly by stating that there is no explicit structure or hierarchy in the $400$ ARC tasks and sampling randomly for $T$ large enough (e.g., $T=10$) should yield results representative of what would be obtained on average on the whole set of $400$ tasks.
<br>

We have to decide on one of the following modalities that will decide on a single associated experiment setting below:
- Train separately on datasets of size $N_1, ..., N_k$ (e.g. $N_1=100, N_2=1000, ..., N=100000$) different samples respectively and write the best performance for each dataset size. We use early stopping or save the best checkpoint.

- Train on a dataset of sufficiently large $N$ (e.g. $N=100000$) samples and report the model performance when $n_1, n_2=..., n_k$ (e.g. $n_1=100, n_2=1000, ..., n_k=100000$) **different** samples have been seen.

- Train on a dataset of fairly large $N$ (e.g. $N=50000$) samples and report the model performance when $n_1, n_2=..., n_k$ (e.g. $n_1=100, n_2=1000, ..., n_k=100000$) samples have been seen.

- Train on a dataset of fairly large $N$ (e.g. $N=50000$) samples and report the number of steps a model has performed to reach its best performance. 
<br>

→ We decide to consider the first modality for the following experiment settings.

<br>

**Experiment setting 1: Varying Number $N$ of Samples**<br>
For each task considered and each $N \in \{100, 1000, 2500, 5000, 10000, 25000, 50000, 100000\}$:
- Train on $N_{train} = N$ samples
- Test on $N_{test}$ samples with the best model checkpoint obtained during training

<br>

**Experiment setting 2: Best Performance across $N$ Different Samples**<br>
Fix a sufficiently large $N=100000$. Consider $n_i \in \{100, 1000, 2500, 5000, 10000, 25000, 50000, 100000\}$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples
- Test on $N_{test}$ samples after the model has seen $n_i$ **different** samples during training

<br>

**Experiment setting 3: Best Performance across $N$ Samples**<br>
Fix a fairly large $N=50000$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples until convergence of the model
- Test on $N_{test}$ samples with the best model checkpoint obtained during training and report how many samples had been seen by that model checkpoint

<br>

**Experiment setting 4: Best Performance across $S$ Steps**<br>
Fix a fairly large $N=50000$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples until convergence of the model
- Test on $N_{test}$ samples with the best model checkpoint obtained during training and report how many steps had been performed by that model checkpoint


## ICL
See later.