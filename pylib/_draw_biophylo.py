from Bio import Phylo as BioPhylo
from matplotlib import pyplot as plt


def draw_biophylo(tree: BioPhylo.BaseTree, **kwargs) -> plt.Axes:
    ax = plt.gca()
    BioPhylo.draw(tree, axes=ax, do_show=False, **kwargs)
    return ax
