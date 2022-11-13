import os
import pandas as pd

# files = os.listdir("./OriginData/")
files = os.listdir("./Data/")
dfs = []
map_label = {}
for file in files:
    if not file.endswith("txt") or file.startswith("ReadMe"):
        continue
    else:
        # label = file.split("_")[2]
        # print(label)
        df = pd.read_csv(os.path.join("./Data/", file), sep="\t", skiprows=3)
        label_code = df["act"].tolist()[0]
        map_label[label_code] = file
        labels = df["act"].unique()
        if len(labels) > 1:
            for label in labels:
                tmp = file.split(".")
                new_file_name = "{}_label_{}.csv".format(tmp[0], label)
                df.loc[df['act']==label].to_csv(os.path.join("./TransformData", new_file_name), index=False)
        else:
            tmp = file.split(".")
            new_file_name = "{}.csv".format(tmp[0])
            df.to_csv(os.path.join("./TransformData", new_file_name), index=False)
        # dfs.append(df)

df_all = pd.concat(dfs)
df_all.to_csv("all_data.csv", index=False)
# df_label = pd.DataFrame({"LabelCode": list(map_label.keys()), "Label": list(map_label.values())})
# df_label.sort_values(by="LabelCode", inplace=True)
# df_label.to_csv("map_label.csv", index=False)