technologyspace
====
This is the code accompanying the paper:
Jeff Alstott, Giorgio Triulzi, Bowen Yan, Jianxi Luo. (2015). "Mapping Technology Space by Normalizing Technology Relatedness Networks." Available at `arXiv:1509.07285 [physics.soc-ph]`__

__ http://arxiv.org/abs/1509.07285

How to Use
====
The code base is organized as a set of `IPython notebooks`__, which are also duplicated as simple Python ``.py`` script files. The only thing you should need to touch directly is the notebook `Manuscript_Code`__ , which walks through all the steps of:

1. organizing the raw empirical data
2. creating technology relatedness networks from the empirical data
3. creating randomized versions of the data and calculating technology relatedness networks from it.
4. comparing the empirical and randomized versions of the networks
5. creating figures for the `manuscript`__, the source code for which is also contained in this repository.

__ http://ipython.org/notebook.html
__ https://github.com/jeffalstott/technologyspace/blob/master/src/Manuscript_Code.ipynb
__ http://arxiv.org/abs/1509.07285

The raw data files we use are too large to host on Github (>100MB), and we are figuring out an external place to host them. Once this is done, the `Manuscript_Code` notebook will automatically download the raw data before processing it.

[More details forthcoming].

Randomization with a cluster
====
This pipeline involves creating thousands of randomized versions of the historical patent data. In order to do this, we employ a computational cluster running the `PBS`__ job scheduling system. Running this code currently assumes you have one of those. If you are lucky enough to be from the future, maybe you have a big enough machine that you can simply create and analyze thousands of randomized versions of the historical patent data using a simple ``for`` loop. We don't yet support that.

__ https://en.wikipedia.org/wiki/Portable_Batch_System


Dependencies
====
- Python 3.x
- `powerlaw`__
- `seaborn`__
- `pyBiRewire`__
- the standard scientific computing Python stack, which we recommend setting up by simply using the `Anaconda Python distributon`__. Relevant packages include:
- - numpy
- - scipy
- - matplotlib

__ https://github.com/jeffalstott/powerlaw
__ http://stanford.edu/~mwaskom/software/seaborn/
__ https://github.com/andreagobbi/pyBiRewire
__ http://docs.continuum.io/anaconda/index

Original Data Files
====
- citing_cited.csv
- PATENT_US_CLASS_SUBCLASSES_1975_2011.csv
- pid_issdate_ipc.csv
- disamb_data_ipc_citations_2.csv
- pnts_multiple_ipcs_76_06_valid_ipc.csv
- patent_ipc_1976_2010.csv
