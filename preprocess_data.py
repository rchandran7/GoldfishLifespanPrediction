import sys
import pandas as pd
import numpy as np

if len(sys.argv) == 3:
    file_input = sys.argv[1]
    file_output = sys.argv[2]
else:
    print("Error: Invalid number of arguments.", file=sys.stderr)

df = pd.read_csv(file_input)
df.drop(['id', 'color'], axis=1)
