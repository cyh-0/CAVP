import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
import torch.nn.functional as F
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def tsne_plotter(encoded_cluster_x, encoded_cluster_y, c_list, num_classes=23, name="example"):
    x = F.normalize(encoded_cluster_x, p=2, dim=1).squeeze().detach().numpy()
    # x = encoded_cluster_x.squeeze().cpu().detach().numpy()
    y = encoded_cluster_y.detach().numpy()

    feat_cols = ['F' + str(i) for i in range(0, x.shape[1])]
    df = pd.DataFrame(x, columns=feat_cols)
    df['label'] = y
    # df['class'] = np.asarray(c_list).reshape(-1,1)

    # pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000, init='random')
    tsne_results = tsne.fit_transform(df[feat_cols].values)
    
    df['X'] = tsne_results[:, 0]
    df['Y'] = tsne_results[:, 1]

    # tmp_center = df.iloc[[-1]]
    df = df[:-1]
    plt.clf()
    plt.figure(figsize=(15,7.5), dpi= 80)
    # sns.set_style("whitegrid")
    num_classes = df['label'].unique().shape[0]
    sns.scatterplot(
        x="X",
        y="Y",
        # z='Z',
        hue="label",
        style="label",
        palette=sns.color_palette(n_colors=num_classes, as_cmap=False),
        # palette=color_dict,
        data=df,
        # markers="o",
        s=100,
        legend=True,
        alpha=1.0,
        edgecolor="black"
    )
    # plt.title("audio", fontsize=34)
    # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2),loc="lower left",
                mode="expand", borderaxespad=0, ncol=4)
    plt.savefig(f"visualisation/{name}.pdf", bbox_inches='tight', transparent="True", pad_inches=0)

    # redraw the canvas
    fig = plt.gcf()
    fig.canvas.draw()
    # convert canvas to image using numpy
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    wandb.log({f"tsne/{name}": wandb.Image(img)})

    # plt.savefig(os.path.join(wandb.run.dir, "media",f"{name}.png"), bbox_inches='tight', transparent="True", pad_inches=0)
    