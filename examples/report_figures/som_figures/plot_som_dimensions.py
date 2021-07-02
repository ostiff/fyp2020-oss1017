"""
SOM dimension analysis
======================

Results of analysis of SOM dimensions on metrics.

"""

import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


sys.path.insert(0, os.path.abspath('.'))

# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'font.size': 18,
# })


mpl.rcParams.update({
    'font.family': 'serif',
    'font.weight': 'light',
    'font.size': 16,
})

NUM_NODE_DATA = {
    'n_nodes': [100,169,225,324,400,625,900,1225,1600,2025,2500],
    'gmm': [0.646658546977072,0.796442496790773,0.720634757044514,0.7738323743976,0.794637531209736,0.400484080730981,0.368259754484194,0.789911646195414,0.377878322923894,0.572371246513618,0.682202543544116],
    'convex': [0.835294117647058,0.852842809364548,0.814814814814814,0.848484848484848,0.818918918918919,0.794893617021276,0.848538011695906,0.810234541577825,0.689935064935065,0.862324393358876,0.875051546391752],
    'concave': [0.401234567901234,0.395833333333333,0.841836734693877,0.366782006920415,0.839335180055401,0.810763888888889,0.321046373365041,0.821799307958477,0.699934123847167,0.875973015049299,0.89047419219471],
    'correlation': [0.662109264610594,0.648898339468736,0.712307268055736,0.69686501138573,0.71459235011473,0.604920069781044,0.572365709607443,0.59152617294072,0.559749099934296,0.638488789338168,0.61715772098528],
    'procrustes': [0.458880684449899,0.466436354803268,0.433866850163302,0.45573989009054,0.445057222226785,0.53203055094161,0.555853784051347,0.515464678685324,0.572550426501338,0.516550232581764,0.543472600943776]
}

SIDES_RATIO_DATA = {
    'ratio': [1,2,3,4.33441144608176,5,5.5,6,6.5,7,0.5,0.75],
    'gmm': [0.406373378996569,0.463352068506144,0.373715273515916,0.3215775114413,0.325909432617165,0.481508449464127,0.397243867424433,0.50556142976656,0.447864767205106,0.471113765549755,0.490774512047504]
}


# %%
# Number of nodes impact on distance metrics
# -------------------------------------------

df = pd.DataFrame(data=NUM_NODE_DATA)


plt.figure()
sns.lineplot(data=df, x="n_nodes", y="correlation", label="Pearson correlation")
sns.lineplot(data=df, x="n_nodes", y="procrustes", label="Procrustes disparity")
plt.legend()
plt.xlabel("Number of nodes")
plt.ylabel("Metric value")
plt.axvline(602, 0, 1, c='r', ls='--')
# plt.savefig("som_n_nodes_dist.pdf", bbox_inches='tight')
plt.show()


# %%
# Impact of ratio of sides on density metrics
# -------------------------------------------

df = pd.DataFrame(data=SIDES_RATIO_DATA)


plt.figure()
sns.lineplot(data=df, x="ratio", y="gmm", label="GMM ratio")
plt.legend()
plt.xlabel("Ratio of sides ($X / Y$)")
plt.ylabel("Metric result")
plt.axvline(4.334411446, 0, 1, c='r', ls='--')
# plt.savefig("som_ratio_density.pdf", bbox_inches='tight')
plt.show()


