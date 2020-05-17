PAMOGK
=====

# Installation

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

First install `pipenv` using the instructions given:

> https://github.com/pypa/pipenv#installation

We are using [pipenv](https://github.com/pypa/pipenv#installation) because of version locking and predictive builds.

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
We also extend default paths with some custom paths.

You can check `pamogk.config.MOSEK_SUPPORTED_PATHS` to see a list of supported paths.

Other option is to set the `MOSEKLM_LICENSE_FILE` environment variable to your license file location.

For more information:

> https://docs.mosek.com/9.2/install/installation.html#setting-up-the-license

## Dependencies
Install dependencies with:
```bash
pipenv install
```

## Required Environment Variables
You can add a `.env` file that will be loaded by `pipenv`

Also please don't forget to export `MOSEKLM_LICENSE_FILE` variable as well if you are using a path other
than `$HOME/mosek/mosek.lic`:

```bash
# required for experiment running
PYTHONPATH=${PYTHONPATH}:.
MOSEKLM_LICENSE_FILE=$MY_MOSEK_LICENSE_FILE_PATH
```

# Folder Structure
- **data:** Data files that are both raw or generated by the project. Keeping intermediate files saves a lot of
 computation time where applicable.
- **experiments:** Experiments done on real data.
- **pamogk:** package root folder
  - **kernels:** Kernel methods to calculate distance between patients. Will try to extend these!
  - **data_processor:** Data processors/converters used to convert different forms of data to usable forms.
   Will try to extend these!
  - **gene_mapper:** Uniprot/Entrez gene mapping matcher for different datasets using different gene IDs. Will try to
   separate to its own package or along with other tools.
  - **pathway_reader:** https://.ndexbio.org CX format, and http://kegg.jp KGML format reader and converters. Will try
   to separate to its own package or along with other tools.


# Experiments
## Running Experiments
You should run experiments either by starting a pipenv environment shell by:
```bash
pipenv shell
```

or through pipenv with:
```bash
pipenv run experiments/pamogk_all_exp_1.py
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
