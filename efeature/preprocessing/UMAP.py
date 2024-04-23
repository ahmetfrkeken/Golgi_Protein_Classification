import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
#current_palette = sns.color_palette()
#sns.palplot(current_palette)
sns.axes_style()
{'axes.axisbelow': True,
 'axes.edgecolor': '.15',
 'axes.facecolor': 'white',
 'axes.grid': False,
 'axes.labelcolor': '.15',
 'axes.linewidth': 1.25,
 'figure.facecolor': 'white',
 'font.family': ['sans-serif'],
 'font.sans-serif': ['Arial',
  'DejaVu Sans',
  'Liberation Sans',
  'Bitstream Vera Sans',
  'sans-serif'],
 'grid.color': '.8',
 'grid.linestyle': '-',
 'image.cmap': 'rocket',
 'legend.frameon': False,
 'legend.numpoints': 1,
 'legend.scatterpoints': 1,
 'lines.solid_capstyle': 'round',
 'text.color': '.15',
 'xtick.color': '.15',
 'xtick.direction': 'out',
 'xtick.major.size': 6.0,
 'xtick.minor.size': 3.0,
 'ytick.color': '.15',
 'ytick.direction': 'out',
 'ytick.major.size': 6.0,
 'ytick.minor.size': 3.0}

def plotUMAP(data,outPNG,n_components=2,min_dist=0.99,n_neighbors=38,metric="euclidean"):#"yule"):"euclidean"
    print("Uniform Manifold Approximation and Projection for Dimension Reduction\nUMAP....")
    #print(data.head(1))
    X=data.iloc[:,1:]
     
     
    reducer = umap.UMAP(random_state=2021,n_components=n_components,min_dist=min_dist,n_neighbors=n_neighbors,metric=metric)
    embedding = reducer.fit_transform(X)
    #print(embedding.shape)
    UMAP_out=pd.DataFrame(embedding,columns=["UMAP_D1","UMAP_D2"])
    UMAP_out["Category"]=data["Class"].values
    #print(UMAP_out.head())
    UMAP_out.index=data.index
    UMAP_out.to_csv(outPNG+".csv")
    print("UMAP data", outPNG+".csv"," is saved!")
    '''
    sns_plot=sns.scatterplot(data = UMAP_out
                ,x = "UMAP_D1"
                ,y = "UMAP_D2"
                ,hue = "Category"
                #,palette="pastel"
                )
    fig = sns_plot.get_figure()
    fig.savefig(outPNG+".png",dpi=100,bbox_inches='tight')
    plt.close()
    print("UMAP_plot ",outPNG+".png"," is saved!")
    '''
    #return fig

