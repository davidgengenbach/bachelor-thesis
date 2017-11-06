EXPORT_DPI = 100
EXPORT_FIG_SIZE = (8, 4)
EXPORT_FIG_SIZE_BIG = (10, 7)
EXPORT_FIG_WIDTH, EXPORT_FIG_HEIGHT = EXPORT_FIG_SIZE
EXPORT_FIG_WIDTH_BIG, EXPORT_FIG_HEIGHT_BIG = EXPORT_FIG_SIZE_BIG

import pandas as pd

pd.options.display.max_rows = 80
pd.options.display.max_columns = 999
pd.options.display.max_colwidth = -1

import seaborn as sns
import matplotlib.pyplot as plt
sns.set('notebook', 'whitegrid', palette = 'deep')
plt.rcParams['figure.figsize'] = EXPORT_FIG_SIZE_BIG
plt.rcParams['figure.dpi'] = EXPORT_DPI