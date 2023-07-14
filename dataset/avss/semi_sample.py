import pandas as pd
import json
import os
from config.avss.config_avsbench_semantics import cfg as avss_cfg

root = avss_cfg.DATA.META_CSV_PATH
with open(avss_cfg.DATA.LABEL_IDX_PATH, 'r') as fr:
    index_table = json.load(fr)

df = pd.read_csv(root, dtype=str)
df["a_obj"] = df["a_obj"].apply(lambda x: x.split("_"))

sub_sample_rate = 2
ratio = 1/sub_sample_rate

semi_df = pd.DataFrame(columns=df.columns)
for key, value in index_table.items():
    # if key == "background":
    #     semi_df = pd.concat([semi_df, curr_train], ignore_index=True)
    # else:
    curr = df[df["a_obj"].apply(lambda x: key in x)]
    org_count = curr.shape[0]   
    curr_train = curr[curr["split"] == "train"]
    # pandas random sample 10% of the curr_train
    curr_train = curr_train.sample(frac=ratio, random_state=1, replace=False)
    # concat to semi_df
    semi_df = pd.concat([semi_df, curr_train], ignore_index=True)
    print("Class {:<20} sampled {}/{}".format(f"[{key}]", curr_train.shape[0], org_count))

semi_df.to_csv(os.path.join(os.path.dirname(root), f"metadata_1-{sub_sample_rate}.csv"), index=False)
a=111