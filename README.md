This code is used for dMRI scheme sampling.

### Setup

1. Clone this repository 
2. Install dependencies
```
pip install -r requirements.txt
```
Note that you will need to acquire a license to use GUROBI for solving discrete problems here. For more information, see:
+ https://pypi.org/project/gurobipy/
+ https://www.gurobi.com/academia/academic-program-and-licenses/
+ https://www.gurobi.com/free-trial/

### Quick-start tutorial 

You can check CLI program with option `-h` for help message.

For a example sampling pipeline, we will first generate a single shell sampling scheme with 30 points and then apply flipping and ordering to it.

1. Generate a scheme
```bash
python ./src/qspace_direction/direction_generation.py --output scheme.txt -n 30
```

2. Flip the resulting scheme
```bash
python ./src/qspace_direction/direction_flip.py --input scheme.txt --output flipped.txt
```

3. Order the resulting scheme
```bash
python ./src/qspace_direction/direction_order.py flipped.txt --output flipped_ordered.txt
```

You can check `flipped_ordered.txt` for the final result. 

### References