# BEFORE-ARC Experiment Settings
The experiment settings are described in each study section below for the BEFORE-ARC data environment.
<br>

There are three problems that can be considered:
- Image (grid) generation
  - A sample contains an input (grid) image and the output is an image obtained by a ground-truth AVR transformation of the input image.

<br>

- Image (grid) generation with demonstrations
  - A sample contains an input (grid) image with also $k=2$ demonstration examples (i.e., input-output pairs representing the AVR transformation to learn) concatenated, and the output is an image obtained by a ground-truth AVR transformation of the input image.

<br>

- Categorical prediction given a set of images: MCQ
  - A sample contains an input (grid) image with also $k$ images concatenated and the output is a categorical value indicating which of the $k$ images is the correct result of the ground-truth AVR transformation.
<br>
<br>

→ We only consider the first problem. Note that the first and the second problem are similar as long as the type of transformation/rule to perform is specified when needed.

A transformation is associated to a task. A transformation is either a primitive (i.e., Translate-Up, Translate-Down, Translate-Right, Translate-Left, Rot-90, Mirror-Horizontal, Mirror-Vertical, Crop-Top, Crop-Bottom, Crop-Left, Crop-Right) or a composition of primitive transformations.
<br>

Constraints of the BEFORE-ARC data environment, maintained uniformly across samples, are:
- Objects' size is smaller than 6 pixels in rows and columns  
- Objects are single colored 
- Objects are 8-connected (not 4-connected)
- Objects are not allowed to touch 
- Demonstration examples are 2 by default (except ICL)
<br>

<br>

The considered varying parameters are:
- Grid size 
- Number of objects (where the def. of an object is w.r.t. the above constraints)
- For ICL: Number of demonstrations


## Systematic Generalization
Consider $T_{elem} = 2$ of the $11$ elementary/primitive transformations. They represent tasks.
<br>

The primitive transformations considered are: Translate-Up, Rot-90. If time allows, we will consider more tasks in addition.
<br>

We justify not selecting the other elementary transformations by assuming that results for the selected transformations would be representative of the results of the ones not selected. We also note resources constraints.

<br>

**Experiment setting 1: Number of Objects Difficulty**<br>
Fix the grid size to $(10,10)$.
<br>

For each task considered:
- Train on $N_{train}$ samples with $nb\_ objects \in \{1,2\}$
- Test on $N_{test}$ samples with $nb\_ objects \in \{3,4\}$

<br>

**Experiment setting 2: Grid Size Difficulty**<br>
Fix the number of objects per image to $2$.
<br>

For each task considered:
- Train on $N_{train}$ samples with $grid\_ size \in \{(10, 10), (11, 10), ..., (12,12)\}$
- Test on $N_{test}$ samples with $grid\_ size \in \{(13, 13), (14, 13), ..., (16,16)\}$


## Compositionality
Consider $T_{elem} = 4$ of the $11$ elementary/primitive transformations and the related composite pair for each combination of depth $1$ and of depth $2$. This yields $T_{composite_1} = \sum_{i = 1}^{T_{elem}} i = \frac{T_{elem}^2 + T_{elem}}{2} = 10$ and $T_{composite_2} = \frac{T_{elem} \cdot (T_{elem} + 1) \cdot (T_{elem} + 2)}{6} = 20$ composite tasks selected, since the rules' order does not matter, we consider all unique tuples between the tasks, and repetitions are allowed.
<br>

The primitive transformations considered are: Translate-Up, Rot-90, Mirror-Hor and Crop-Top. If time allows, we will consider more tasks in addition.
<br>

Note that "depth $1$" means that two primitive transformations are composed. Hence, the composition of two primitive transformations is said to be of depth $1$. Consequently, the composition of a composite transformation, resulting from the composition of two primitive transformations, with a primitive transformation is said to be of depth $2$. Essentially, a primitive transformation has no depth, and the depth of a composition is the number of times that a primitive transformation is applied to an initial transformation in order to obtain the resulting composite transformation.
<br>

When fixing the variable parameters, we justify our choice of $2$ for the number of objects and $(10,10)$ for the grid size by stating that a reasonably small and fixed cardinality is enough for the experiments to highlight meaningful results. We also note the resources constraints. 
<br>

<br>

**Experiment setting 1: Composition from Elementary Tasks to Composite**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

For each pair of elementary tasks in the set of tasks that are elementary tasks making up their associated composite task:
- Train on $N_{train}$ samples from the relevant elementary tasks
- Test on $N_{test}$ samples from the associated composite task

<br>

**Experiment setting 2: Decomposition from Composite to Elementary Tasks**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

For each composite task in the set of tasks that are composite tasks associated with the composition of two elementary tasks:
- Train on $N_{train}$ samples from the composite task
- Test on $N_{test}$ samples from the associated elementary tasks

<br>


**Experiment setting 3: From Composite Tasks to Unseen Composite**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

For each composite task that is the depth $1$ composition of two elementary transformations:
- Train on $N_{train}$ samples from all the relevant composite tasks except one
- Test on $N_{test}$ samples from the relevant composite task not seen during training

<br>

**Experiment setting 4: From Composite Tasks and Restricted Elementary Tasks to Unseen Composite**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

For each elementary task and composite task that is the depth $1$ composition of two elementary transformations:
- Train on $N_{train}$ samples from all the relevant tasks except one composite task and the elementary tasks composing it
- Test on $N_{test}$ samples from the relevant composite task not seen during training

<br>

**Experiment setting 5: From Composite Tasks to Deeper Composite Tasks**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.

For each composite task that is the depth $1$ or depth $2$ of the same elementary transformations:
- Train on $N_{train}$ samples from the relevant composite tasks of depth $1$
- Test on $N_{test}$ samples from the relevant composite tasks of depth $2$ (none were seen during training)

Note that the Experiment Setting 5 can also be seen as a Systematic Generalization study, but in the context of Compositionality.

## Sample-Efficiency
Consider $T_{elem} = 4$ of the $11$ elementary/primitive transformations. They represent tasks. 
<br>

The primitive transformations considered are: Translate-Up, Rot-90, Mirror-Hor and Crop-Top. If time allows, we will consider more tasks in addition.
<br>

We justify not selecting the other elementary transformations by assuming that results for the selected transformations would directly transfer to the not selected ones, as well as noting resources constraints.
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
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

For each task considered and each $N \in \{100, 1000, 2500, 5000, 10000, 25000, 50000, 100000\}$:
- Train on $N_{train} = N$ samples
- Test on $N_{test}$ samples with the best model checkpoint obtained during training

<br>

**Experiment setting 2: Best Performance across $N$ Different Samples**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

Fix a sufficiently large $N=100000$. Consider $n_i \in \{100, 1000, 2500, 5000, 10000, 25000, 50000, 100000\}$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples
- Test on $N_{test}$ samples after the model has seen $n_i$ **different** samples during training

<br>

**Experiment setting 3: Best Performance across $N$ Samples**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

Fix a fairly large $N=50000$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples until convergence of the model
- Test on $N_{test}$ samples with the best model checkpoint obtained during training and report how many samples had been seen by that model checkpoint

<br>

**Experiment setting 4: Best Performance across $S$ Steps**<br>
Fix the number of objects to $2$ and the grid size to $(10,10)$.
<br>

Fix a fairly large $N=50000$.
<br>

For each task considered:
- Train on $N_{train} = N$ samples until convergence of the model
- Test on $N_{test}$ samples with the best model checkpoint obtained during training and report how many steps had been performed by that model checkpoint



## ICL
See later.