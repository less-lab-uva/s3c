# Scripts for Exploratory Work
This folder contains scripts for the exploratory work and is organized as follows:
* `dataset_csv/`
  * Contains a CSV file for each of the open-source datasets explored. Each CSV contains a decomposition of the graph used to generate the statistics shown in the right half of Table 3.
* `exploratory_work.py`
  * Consumes the CSV files from `dataset_csv/` to generate the right half of Table 3.
* `rs2v_times.ipynb`
  * Jupyter Notebook that was used to parse the original logs from clustering the open-source datasets. This information is displayed in Table 4.
## Running the script
The script must be run from within this folder:

```bash
cd s3c/exploratory_work
python3 exploratory_work.py
```