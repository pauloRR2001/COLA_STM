from functions.orekit import propagate_segment
import numpy as np
print(propagate_segment(np.array([6708.137,0,0,0,7.7,0]),600,300,area_m2=.65)[1][-1])
