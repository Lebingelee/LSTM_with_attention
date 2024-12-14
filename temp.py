import tool
import pandas as pd

df_val = pd.read_csv('./data/val.csv')
val = tool.CSV(df_val)
val.rm_product_smaller_nums(30)
df_val = val.csv
re_val = tool.CSV(df_val)


ids = val.get_product_id()
re_ids = re_val.get_product_id()
for key,data in ids.items():
    print(key,data)

for key,data in re_ids.items():
    print(key,data)