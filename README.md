## Description

This code is used to optimize single or multiple shell uniform sampling schemes in diffusion MRI via **continuous optimization** and **discrete optimization**.

*   **Continuous optimization**: generate a single or multiple shell uniform sampling scheme in the continuous sphere $\mathbb{S}^2$ with the antipodal symmetry constraint.
*   **Discrete optimization** includes:
    *   **Polarity optimization (P-P)** : optimize the polarity of an existing single or multiple shell sampling scheme by flipping some samples, such that the flipped scheme is uniform in the sphere without the antipodal symmetry constraint. It is useful for reducing eddy current in dMRI.
    *   **Ordering optimization (P-O)**: optimize the ordering of an existing single or multiple shell sampling scheme, such that every partial scanned sample set is nearly uniform.
    *   **Uniform subsampling optimization**: extract a subsampling subset from an existing single or multiple shell sampling scheme, such that the subsampled single or multiple shell scheme is nearly uniform.

## Install

1. Clone this repository 
```
git clone https://github.com/DiffusionMRITool/spherical_uniform_sampling.git
```
2. Install dependencies
```
pip install -r requirements.txt
```
Note that you will need to acquire a license to use GUROBI for solving discrete problems here. For more information, please see:
+ https://pypi.org/project/gurobipy
+ https://www.gurobi.com/academia/academic-program-and-licenses
+ https://www.gurobi.com/free-trial
3. Install package
```
pip install .
```

## Quick-start tutorial 

You can check CLI program with option `-h` for help message.

### Generate a single shell sampling scheme

For an example of generating a single shell uniform sampling, we will first generate a scheme with 30 points using continuous optimization and then apply polarity optimization and order optimization to it. 

This can be done by simply invoke:
```bash
direction_generation.py --output grad_flipped_ordered.txt -n 30
```

Alternately, it is equivalent to following step-by-step instructions.
1. Generate a sampling scheme via continuous optimization.
```bash
direction_continous_optimization.py --output grad.txt -n 30
```

2. Optimize the polarity of the resulting scheme.
```bash
direction_flip.py --input grad.txt --output grad_flipped.txt
```

3. Optimize the ordering of the resulting scheme
```bash
direction_order.py grad_flipped.txt --output grad_flipped_ordered.txt
```

You can check `grad_flipped_ordered.txt` for the final result. 

### Generate a multiple shell sampling scheme

For an example of a multiple shell sampling pipeline, we will first generate a scheme with $30\times 3$ points using continuous optimization and then apply polarity optimization and order optimization to it. 

This can be done by simply invoke:
```bash
direction_generation.py --output grad_flipped_ordered.txt -n 30,30,30 --bval 1000,2000,3000
```

Alternately, it is equivalent to following step-by-step instructions

1. Generate a multiple shell sampling scheme
```bash
direction_continous_optimization.py --output grad.txt -n 30,30,30
```

2. Optimize the polarity of the resulting schemes
```bash
direction_flip.py --input grad_shell0.txt,grad_shell1.txt,grad_shell2.txt --output grad_flipped.txt 
```

3. Optimize the polarity of the resulting schemes
We need to concatenate 3 shells to make a bvec file.
```bash
combine_bvec_bval.py grad_flipped_shell0.txt,grad_flipped_shell1.txt,grad_flipped_shell2.txt 1000,2000,3000 --output grad_combine.txt
```

Finally we run our ordering script.
```bash
direction_order.py grad_combine_bvec.txt grad_combine_bval.txt --output grad_flipped_ordered.txt
```

### Subsample example

**Single subset from single set problem (P-D-SS)**: if you already have a single shell sampling scheme `grad.txt` (or you may generate one using methods above), you can uniformly extract a single subset of points from it.

```bash
direction_subsample.py --input grad.txt --output grad_subsample.txt -n 30
```

**Multiple subsets from single set problem (P-D-MS)**: you can uniformly extract multiple subsets of points from a single shell scheme.

```bash
direction_subsample.py --input grad.txt --output grad_subsample.txt -n 10,10,10
```

**Multiple subsets from multiple sets problem (P-D-MM)**: given a multiple shell sampling scheme, e.g. the HCP scheme, you can uniformly extract multiple subsets of points from it.

```bash
direction_subsample.py --input grad_b1000.txt,grad_b2000.txt,grad_b3000.txt --output grad_subsample.txt -n 30,30,30
```

## License
This project is licensed under the [LICENSE](LICENSE).