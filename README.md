# Ultrasound-Report-Generation

## data and code preparation

how to run:


first, git clone this repo

secondly, download USData from origin repo.

then organize your code as followed:

```
ultrasound_report_gen/
  gitrepo/
  data/
```

prepare a `.config.yaml` file as followed:
```
HOME: "path to parent of ultrasound_report_gen"
REPO: "the repo name"
```

## Requirements

```
conda create -n urg python=3.10 -y
cd gitrepo
pip install -r requirements.txt
```

## train

```py
python KMVE_RG/my_main.py --debug 10 
```

wandb support code is under 
`Ultrasound-Report-Generation/wandb_urg.py`