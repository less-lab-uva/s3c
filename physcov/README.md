# Comparison with PhysCov
The parameterization used can be found in the [`RRS_distributions.py`](./RRS_distributions.py) file in this folder, which replaces the same folder in the original PhysCov repository.
This parameterization increases the precision of the geometric vectorization by increasing the number discrete points considered.

The RSS vectors acquired from PhysCov are available in the `s3c/study_data/physcov/` folder. The `rss_v2_dict.json` file corresponds to the new parameterization Ψ<sup>*</sup>. The JSON files contain dictionaries where the keys are 1-10 for Ψ<sub>1</sub>-Ψ<sub>10</sub>, though only Ψ<sub>10</sub> is utilized by the pipeline.