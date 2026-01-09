# ch15_ex07_sequence.py
# ch15.4: sequential pattern mining

# NOTE: This Python code uses different algorithm called SPADE.
#       The results should be the same to examples in the book,
#       but underlying algorithms and computational steps 
#       are different from AprioriAll or AprioriSome.
#       For SPADE algorithm, please see the paper below:
#       Zaki, M. J. (2001). SPADE: An efficient algorithm for 
#         mining frequent sequences. Machine learning, 42, 31-60.

# load package
import pandas as pd
from spmf import SPADE

# read/construct specifically formatted data
df = pd.DataFrame({
    'ID': ['S1']*2 + ['S2']*6 + ['S3']*3 + ['S4']*4 + ['S5']*1,
    'Time Points': [0, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 6, 7, 7, 8, 9],
    'Items': ['a', 'b', 'c', 'd', 'a', 'e', 'f', 'g', 'a', 'h', 'g', 'a', 'e', 'g', 'b', 'b'],
    })

# find frequent sequences
spade = SPADE(min_support=0.4)
output = spade.run_pandas(df)
print(output)
