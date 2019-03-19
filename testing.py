import numpy as np
import pandas as pd

data = np.array([['','col_1','col_2'],
                ['Row1',1,2],
                ['Row2',3,4],
                ['Row3',1,2],
                ['Row4',3,4],
                ['Row5',1,2],
                ['Row6',3,4],])
      
df =  pd.DataFrame(data=data[1:,1:],
                  index=data[1:,0],
                  columns=data[0,1:])         
df['col_3'] = df.apply(lambda x: (int(x.col_2) + int(x.col_1)), axis=1)
print(data)
print(df)
