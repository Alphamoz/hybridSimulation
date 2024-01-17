import pandas as pd
data = pd.read_csv('../dataglide23Nov.csv')
# data = data.drop(columns=['Index'])
data['Time'] = data['Time'].round(1)
new_data = data[data['Time']%1 == 0]
print(new_data)

new_data.to_csv("downsampled_data3.csv", index=False)
