<img src="https://raw.githubusercontent.com/kriegeskorte-lab/controversial_stimuli_tutorial/main/figures/decorrelated_optim_4_models_Persian_cat_by_Weimaraner.png" height=600>

# **Synthesizing Controversial Stimuli (a tutorial with PyTorch)**

This is a tutorial on how to generate controversial stimuli for ImageNet classifiers using [PyTorch](https://pytorch.org/). More details on the method of controversial stimuli can be found in our [paper](https://www.pnas.org/cgi/doi/10.1073/pnas.1912334117):

T. Golan, P. C. Raju, N. Kriegeskorte, Controversial stimuli: Pitting neural networks against each other as models of human cognition. Proceedings of the National Academy of Sciences 117, 29330–29337 (2020). DOI: 10.1073/pnas.1912334117

The tutorial is contained in a Jupyter notebook:\
[Synthesizing_Controversial_Stimuli_Tutorial.ipynb](https://github.com/kriegeskorte-lab/controversial_stimuli_tutorial/blob/main/Synthesizing_Controversial_Stimuli_Tutorial.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kriegeskorte-lab/controversial_stimuli_tutorial/blob/main/Synthesizing_Controversial_Stimuli_Tutorial.ipynb) [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/kriegeskorte-lab/controversial_stimuli_tutorial/blob/main/Synthesizing_Controversial_Stimuli_Tutorial.ipynb)

#### Installation:
If you want to run the notebook or the Python code on your own machine (i.e., not on Google Colab), you would have to install the dependencies. The main non-trivial dependency is PyTorch 1.6 (≥1.7 is not supported by Kornia 0.4). On most systems, this should work:
```
git clone https://github.com/kriegeskorte-lab/controversial_stimuli_tutorial
cd controversial_stimuli_tutorial
conda create -n contro_stim_env python==3.7
conda activate contro_stim_env
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch 
pip install -r requirements.txt
conda install -c conda-forge jupyterlab
jupyter lab Synthesizing_Controversial_Stimuli_Tutorial.ipynb
```
#### Batch stimulus synthesis
This repository also includes Python code for synthesizing controversial stimuli in a batch. 
After installing the dependencies, you can synthesize all of the images that appear in the figure above by running:
```
python batch_optimize.py --experiments cat_vs_dog --optimization_methods CPPN --max_steps=1000
```
To synthesize all of the images that appear in the tutorial figures, run `python batch_optimize.py` without arguments (warning: this takes many hours if not done over a GPU cluster). This code is compatible with being run by multiple workers in parallel (it synchronizes tasks through 0-byte file flags). However, let a single worker run alone for a couple of minutes so it can download the necessary DNNs without competition.

#### Citation
Please cite ([Golan, Raju, & Kriegeskorte, 2020](https://www.pnas.org/cgi/doi/10.1073/pnas.1912334117)) if you build upon our method in your research ([bibtex link](https://www.pnas.org/highwire/citation/962270/bibtext)). If want to cite this tutorial directly, you may do so using the following reference:
```bibtex
@software{Golan_2020_Synthesizing,
  author       = {Tal Golan and
                  Nikolaus Kriegeskorte},
  title        = {{Synthesizing Controversial Stimuli (a tutorial with PyTorch)}},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.4291135},
  url          = {https://doi.org/10.5281/zenodo.4291135}
}
```
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4291135.svg)](https://doi.org/10.5281/zenodo.4291135)
