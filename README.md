PAMOGK
=====
# Usage
pamogk is released as a python dependency on [PyPI](https://pypi.org/project/pamogk/)

If you will use the framework to construct your own experiments you can install it as a python dependency depending on your environment:

```bash
pip install pamogk # global package
pipenv add pamogk # pipenv
poetry add pamogk # poetry
```

You can check out the experiments folder for how to construct your own pipeline. `ood_` prefix refers to _out of date_ experiments, you can use them for reference but they might be broken.

# Development Setup

## Getting the Code Base
You can either checkout the code base through git:
```bash
git clone https://github.com/tastanlab/pamogk.git
```
or you can download the master branch from:
> https://github.com/tastanlab/pamogk/archive/master.zip

## Python Environment
Project mainly uses python, but some of the plot generation codes use matlab or output html files (plotly.js). Also some
 libraries might be dependent on operating system. In our tests we used Debian 12.04LTS and macOS Catalina.

First install `poetry` using the instructions given:

> https://python-poetry.org/docs/#installation

We are using [poetry](https://python-poetry.org/docs/) because of version locking, predictive builds, and its speed vs [pipenv](https://github.com/pypa/pipenv#installation).

## Setting up MOSEK
We are using [MOSEK Optimizer API](https://docs.mosek.com/9.1/pythonapi/index.html) for optimizing view weights, and in
order to use MOSEK you will need a license file. MOSEK provides free certificates for academic purposes and free trial
certificates for Commercial usages. You can see check the link below for information on how to acquire an academic license:

> https://www.mosek.com/products/academic-licenses/

### License File Placement
You can either place the license file under user's home directory in a folder named `mosek` e.g:
```bash
$HOME/mosek/mosec.lic # *nix/macOS users
%USERPROFILE%\mosek\mosek.lic # windows users
```
Other option is to set the `MOSEKLM_LICENSE_FILE` environment variable to your license file location.

We also look for the license file paths in some custom paths (by overwriting `MOSEKLM_LICENSE_FILE` if not present).
You can check `pamogk.config.MOSEK_SUPPORTED_PATHS` to see a list of supported paths.

For more information:

> https://docs.mosek.com/9.2/install/installation.html#setting-up-the-license

## Dependencies
Install dependencies with:
```bash
poetry install
```

## Required Environment Variables
You should add project root to the python path with (where `PROJECT_PATH` is the absolute path of the project):
```bash
export PYTHONPATH="${PYTHONPATH}:${PROJECT_PATH}"
```

Also please don't forget to export `MOSEKLM_LICENSE_FILE` variable as well if you are using a path other than `$HOME/mosek/mosek.lic`:

```bash
# required for experiment running
PYTHONPATH=${PYTHONPATH}:.
MOSEKLM_LICENSE_FILE=$MY_MOSEK_LICENSE_FILE_PATH
```

# Folder Structure
- **pamogk:** package root folder
  - **kernels:** Kernel methods to calculate distance between patients. Will try to extend these!
  - **data_processor:** Data processors/converters used to convert different forms of data to usable forms.
   Will try to extend these!
  - **gene_mapper:** Uniprot/Entrez gene mapping matcher for different datasets using different gene IDs. Will try to
   separate to its own package or along with other tools.
  - **pathway_reader:** https://.ndexbio.org CX format, and http://kegg.jp KGML format reader and converters. Will try
   to separate to its own package or along with other tools.
- **data:** Data files that are both raw or generated by the project. Keeping intermediate files saves a lot of
 computation time where applicable. (not included in package)
- **experiments:** Experiments done on real data. (not included in package)


# Experiments
## Running Experiments
You should run experiments either by starting a virtual environment shell by:
```bash
poetry shell
```

or through poetry with:
```bash
poetry run python experiments/pamogk_all_exp_1.py
```
This ensures that experiments are run with correct environment.

## Experiment Parameters
Experiments are code entry points that have arguments that change per experiment. You can see their arguments by
 running them with `--help` parameter. e.g:
```bash
./experiments/pamogk_all_exp_1.py --help
```
and you should see help as such:
```
usage: pamogk_all_exp_1.py [-h] [--rs-patient-data file-path]
                           [--rp-patient-data file-path]
                           [--som-patient-data file-path]

Run PAMOGK-mut algorithms on pathways

optional arguments:
  -h, --help            show this help message and exit
  --rs-patient-data file-path, -rs file-path
                        rnaseq pathway ID list
  --rp-patient-data file-path, -rp file-path
                        rppa pathway ID list
  --som-patient-data file-path, -s file-path
                        som mut pathway ID list
```
