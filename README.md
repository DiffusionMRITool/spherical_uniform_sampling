This code is used for optimizing dMRI sampling schemes.

## Setup

1. Clone this repository 
2. Install dependencies
```
pip install -r requirements.txt
```
Note that you will need to acquire a license to use GUROBI for solving discrete problems here. For more information, see:
+ https://pypi.org/project/gurobipy/
+ https://www.gurobi.com/academia/academic-program-and-licenses/
+ https://www.gurobi.com/free-trial/
3. Install package
```
pip install .
```

## Quick-start tutorial 

You can check CLI program with option `-h` for help message.

For a example single shell sampling pipeline, we will first generate a scheme with 30 points and then apply flipping and ordering to it.

This can be done by simply invoke
```bash
direction_generation.py --output flipped_ordered.txt -n 30
```

Alternately, it is equivalent to following step-by-step instructions
1. Generate a sampling scheme
```bash
direction_continous_optimization.py --output scheme.txt -n 30
```

2. Optimize the polarity of the resulting scheme
```bash
direction_flip.py --input scheme.txt --output flipped.txt
```

3. Optimize the ordering of the resulting scheme
```bash
direction_order.py flipped.txt --output flipped_ordered.txt
```

You can check `flipped_ordered.txt` for the final result. 

For a example multiple shell sampling pipeline, we will first generate a scheme with $90\times 3$ points and then apply flipping and ordering to it.

This can be done by simply invoke
```bash
direction_generation.py --output scheme.txt -n 90,90,90 --bval 1000,2000,3000
```

Alternately, it is equivalent to following step-by-step instructions

1. Generate a multiple shell sampling scheme
```bash
direction_continous_optimization.py --output flipped_ordered.txt -n 90,90,90
```

2. Optimize the polarity of the resulting schemes
```bash
direction_flip.py --input scheme_shell0.txt,scheme_shell1.txt,scheme_shell2.txt --output flipped.txt 
```

3. Optimize the polarity of the resulting schemes
We need to concatenate 3 shells to make a bvec file.
```bash
combine_bvec_bval.py flipped_shell0.txt,flipped_shell1.txt,flipped_shell2.txt 1000,2000,3000 --output combine.txt
```

Finally we run our ordering script.
```bash
direction_order.py combine_bvec.txt combine_bval.txt --output flipped_ordered.txt
```

### License
This project is licensed under the [CC-BY-NC-4.0](LICENSE.md).