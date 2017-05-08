technologyspace
====
This is the data and code accompanying the paper:
Jeff Alstott, Giorgio Triulzi, Bowen Yan, Jianxi Luo. (2017). "Mapping Technology Space by Normalizing Technology Relatedness Networks." Scientometrics. 110(1):443–479. Available at `Scientometrics`__ or on arXiv at `arXiv:1509.07285 [physics.soc-ph]`__

__ https://link.springer.com/article/10.1007/s11192-016-2107-y
__ http://arxiv.org/abs/1509.07285

Data: The Technology Space
====
The data describing the technology space is available in `this Dropbox directory`__. The simplest data are the network as caculated with data from 1975-2010, which is recorded as CSVs. These are in three folders depending on what classification system you're using:
- "USPC" (the United States Patent Classification System) 
- "IPC" (the International Patent Classification System, at the 3-digit level)
- "IPC4" (the International Patent Classification System, at the 4-digit level)

Each CSV is just a rectangular array (number of classes * number of classes), with the values between the strength of the relatedness between each class. Short names for each class in the IPC and IPC4 systems are included as separate text files.

__ https://www.dropbox.com/sh/lfxay4ztvn21ztf/AAD6aii9HR1cLVhzGywHJVpZa?dl=0


Code
====
In this repository are the code to perform the analyses and create the figures in the paper.

How to Use
---
The code base is organized as a set of `IPython notebooks`__, which are also duplicated as simple Python ``.py`` script files. The only thing you should need to touch directly is the notebook `Manuscript_Code`__ , which walks through all the steps of:

1. organizing the raw empirical data
2. creating technology relatedness networks from the empirical data
3. creating randomized versions of the data and calculating technology relatedness networks from it.
4. comparing the empirical and randomized versions of the networks
5. creating figures for the `manuscript`__, the source code for which is also contained in this repository.

__ http://ipython.org/notebook.html
__ https://github.com/jeffalstott/technologyspace/blob/master/src/Manuscript_Code.ipynb
__ http://arxiv.org/abs/1509.07285

The data files we use are too large to host on Github (>100MB), and so are hosted as a 3.6GB ZIP file on Dropbox `here`__. Just download and unzip it in ``technologyspace`` folder, alongside ``src`` and ``manuscript``. This file contains both the raw input data and several intermediate data files produced by the pipeline.

__ https://www.dropbox.com/s/of3gg3gwv0bhtop/data.zip?dl=0

Randomization with a cluster
---
This pipeline involves creating thousands of randomized versions of the historical patent data. In order to do this, we employ a computational cluster running the `PBS`__ job scheduling system. Running this code currently assumes you have one of those. If you are lucky enough to be from the future, maybe you have a big enough machine that you can simply create and analyze thousands of randomized versions of the historical patent data using a simple ``for`` loop. We don't yet support that.

__ https://en.wikipedia.org/wiki/Portable_Batch_System


Dependencies
---
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
---
- citing_cited.csv
- PATENT_US_CLASS_SUBCLASSES_1975_2011.csv
- pid_issdate_ipc.csv
- disamb_data_ipc_citations_2.csv
- pnts_multiple_ipcs_76_06_valid_ipc.csv
- patent_ipc_1976_2010.csv

Contact
====
Please contact the authors if you have questions/comments/concerns/stories:
jeffalstott at gmail
