This code is used for optimizing dMRI sampling schemes.

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
3. Install package
```
python setup.py install
```

### Quick-start tutorial 

You can check CLI program with option `-h` for help message.

For a example single shell sampling pipeline, we will first generate a scheme with 30 points and then apply flipping and ordering to it.

1. Generate a sampling scheme
```bash
python -m qspace_direction.direction_generation --output scheme.txt -n 30
```

2. Optimize the polarity of the resulting scheme
```bash
python -m qspace_direction.direction_flip --input scheme.txt --output flipped.txt
```

3. Optimize the ordering of the resulting scheme
```bash
python -m qspace_direction.direction_order flipped.txt --output flipped_ordered.txt
```

You can check `flipped_ordered.txt` for the final result. 

For a example multiple shell sampling pipeline, we will first generate a scheme with $90\times 3$ points and then apply flipping and ordering to it.

1. Generate a multiple shell sampling scheme
```bash
python -m qspace_direction.direction_generation --output scheme.txt -n 90,90,90
```

2. Optimize the polarity of the resulting schemes
```bash
python -m qspace_direction.direction_flip --input scheme_shell0.txt,scheme_shell1.txt,scheme_shell2.txt --output flipped.txt 
```

3. Optimize the polarity of the resulting schemes
We need to concatenate 3 shells to make a bvec file.
```bash
cat flipped_shell0.txt flipped_shell0.txt flipped_shell0.txt > bvec.txt
```
Then a bval file is needed, here we create one with bvals 1000, 2000, 3000 for each shell.
```bash
perl -e '$count=90; while ($count>0) { print "1000\n"; $count--; }
         $count=90; while ($count>0) { print "2000\n"; $count--; }
         $count=90; while ($count>0) { print "3000\n"; $count--; }
' > bval.txt
```
Finally we run our ordering script.
```bash
python -m qspace_direction.direction_order bvec.txt bval.txt --output flipped_ordered.txt
```

### License
This project is licensed under the [MIT License](LICENSE).