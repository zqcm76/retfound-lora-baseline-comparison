import sys; sys.path.insert(0,'phase1')
from masks import sanity_check_encoding
from PIL import Image; import numpy as np, pandas as pd
p = pd.read_csv('data/data_all.csv')['mask_path'].iloc[0]
print(sanity_check_encoding(np.asarray(Image.open(p).convert('L'))))
