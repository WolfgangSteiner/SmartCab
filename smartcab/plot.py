import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data_explore_first_defensive_2.csv")
print df

df2 =  df.pivot(index='gamma', columns='alpha', values='success')
df3 =  df.pivot(index='gamma', columns='alpha', values='penalty_per_step')
df4 =  df.pivot(index='gamma', columns='alpha', values='efficiency')

fig, ax = plt.subplots()
plt.title('success rate')
ax = sns.heatmap(df2, annot=True, cbar=False)
#plt.tight_layout()
fig.set_size_inches(5,4.5)
fig.savefig('fig/success_rate.pdf')
#sns.plt.show()

fig, ax = plt.subplots()
plt.title('penalty per step')
ax = sns.heatmap(df3, annot=True, cbar=False)
#plt.tight_layout()
fig.set_size_inches(5,4.5)
fig.savefig('fig/penalty.pdf')

fig, ax = plt.subplots()
plt.title('efficiency')
ax = sns.heatmap(df4, annot=True, cbar=False)
#plt.tight_layout()
fig.set_size_inches(5,4.5)
fig.savefig('fig/efficiency.pdf')
