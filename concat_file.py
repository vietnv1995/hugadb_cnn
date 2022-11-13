import os
import pandas as pd

files = os.listdir("./TransformData/")
nrow_split = 100
dfs = []
for file in files:
    df = pd.read_csv(os.path.join("./TransformData/", file))
    # print("File name: {}. Number rows: {}".format(file, df.shape[0]))
    nrow = df.shape[0]
    # if nrow >= 500:
    #     print(file)
    person = file.split("_")[-2]
    if person == "label":
        person = file.split("_")[-4]
    person = int(person)
    print(person)
    if nrow < nrow_split:
        # continue
        print("File name: {}. Number rows: {}".format(file, df.shape[0]))
    else:
        batch = nrow//nrow_split
        df = df[0:batch*nrow_split]
        print(df.shape[0])
        df["file_name"] = ["{}.csv".format(person)]*df.shape[0]
        dfs.append(df)

df_all = pd.concat(dfs)
df_all.to_csv("all_data_transformed.csv", index=False)