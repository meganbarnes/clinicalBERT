# Running Fine-tuning

### Create conda environment
`conda create --name <env> --file requirements.txt`

### For MedNLI

Change main script to `run_distil_classifier.py` and set `--task_name mednli`

### For GOC

Change main script to `run_doc_distil_classifier.py` and set `--task_name goc`

#### Run script
`./run_classifier.sh`
