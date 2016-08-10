import pandas as pd
import seaborn as sns

df = pd.read_csv("data.csv")
print df
df2 =  df.pivot(index='gamma', columns='alpha', values='penalty_per_step')
sns.heatmap(df2, annot=True)
sns.plt.show()
