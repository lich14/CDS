# Run an experiment

``` python 
cd QPLEX-master-SC2/pymarl-master
python3 src/main.py --config=qplex_qatten_sc2 --env-config=sc2_corridor

```

## Used maps in the paper
--env-config=
1.  sc2_corridor
2.  sc2_MMM2
3.  sc2_6h_vs_8z
4.  sc2_3s5z_vs_3s6z


##

To run our method with QPLEX, one should add following codes in the __init__() function of smac/env/starcraft2/starcraft2.py. Meanwhile, one should add one more function called get_unit_dim(self) which returns self.unit_dim.

``` 
self.unit_dim = 4 + self.shield_bits_ally + self.unit_type_bits
``` 
