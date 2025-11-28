# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Create a sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 30, 22, 35],
    'Salary': [50000, 60000, 45000, 70000]
}

df = pd.DataFrame(data)

# %%
import numpy as np
from shared_functions.FEM2D import FEM2D_frame
# ----- NODES -----
nodes = [
    [0, 0],   # node 0
    [0, 3],   # node 1
    [5, 3],   # node 2
    [5, 0]    # node 3
]
# ----- ELEMENT CONNECTIVITY -----
elements = [
    [0, 1],   # left column
    [1, 2],   # top beam
    [2, 3]    # right column
]
# ----- ELEMENT PROPERTIES -----
elem_props = [
    {'type':'beam', 'A':0.02, 'I':8e-5, 'E':210e6, 'q':0},   # element 0
    {'type':'beam', 'A':0.02, 'I':8e-5, 'E':210e6, 'q':2},   # element 1 (top beam with load)
    {'type':'beam', 'A':0.02, 'I':8e-5, 'E':210e6, 'q':0}    # element 2
]
# ----- POINT LOADS -----
loads = []   # no point loads
# ----- SUPPORTS -----
constraints = [1, 9, 10]   # left roller: 1 ; right pinned: vertical DOF 9,10
# ----- RUN FEM -----
u, reactions, axial, shear, moments = FEM2D_frame(nodes, elements, elem_props, loads, constraints)


