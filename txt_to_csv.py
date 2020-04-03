import pandas as pd

df = pd.read_csv("data_banknote_authentication.txt",delimiter=',')
df.to_csv('data_banknote_authentication.csv', index = False)
