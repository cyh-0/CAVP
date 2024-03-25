import os
import pandas as pd
from glob import glob
from dataset.vpo_mono.single_source.av_datasets import AudioVisualDataset


def get_csv(args):
    if args.dataset_name == "vgg10k" or args.dataset_name == "vggss":
        fn = os.path.join(args.root_dataset_dir, "csv", "vggsound_10k.csv")

    elif args.dataset_name == "vggsound_instruments":
        fn = os.path.join(args.root_dataset_dir, "csv", "vggsound_instruments_train.csv")
    else:
        raise FileNotFoundError

    df_vgg = pd.read_csv(
        fn,
        header=None,
        names=["key", "st", "label", "split"],
        dtype=str,
    )

    df_vgg["st"] = df_vgg["st"].str.zfill(6)
    df_vgg["file"] = df_vgg["key"] + "_" + df_vgg["st"]
    out_df = df_vgg[df_vgg["split"] == "train"]
    # avail_files = set(out_df["file"].tolist())
    return out_df


def get_train_dataset(args):
    train_df = get_csv(args)
    file_name = list(train_df["file"])
    all_bboxes = dict()
    for item in file_name:
        all_bboxes.update({item: []})

    return AudioVisualDataset(
        args,
        mode="train",
        data_path=args.train_data_path,
        dataframe=train_df)


def get_test_files(args):
    if args.dataset_name == "vggss":
        df_test = pd.read_json("./metadata/vggss.json")
        # testset = set(df_test.file)
    elif args.dataset_name == "vggsound_instruments":
        fn = os.path.join(args.root_dataset_dir, "csv", "vggsound_instruments_test.csv")
        df_test = pd.read_csv(fn, index_col="sample_index")
        for index, row in df_test.iterrows():
            file_list = glob(os.path.join(args.test_data_path, "Masks", row["current_frame_path"][:-4]
                                          + ".pkl"))
            if len(file_list) == 0:
                df_test.drop(index, inplace=True)
    else:
        raise NotImplementedError
    return df_test


def get_test_dataset(args):
    test_df = get_test_files(args)
    return AudioVisualDataset(
        args=args,
        mode="test",
        data_path=args.test_data_path,
        dataframe=test_df,
    )
