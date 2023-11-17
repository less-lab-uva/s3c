# Comparison with PhysCov
We compare against [PhysCov](https://github.com/hildebrandt-carl/PhysicalCoverage) as a baseline in Section 4.2 using ground-truth LiDAR data that was gathered from CARLA. 
Specifically, we compare against the Ψ<sub>10</sub> parameterization outlined in PhysCov as it is the richest abstraction discussed in their study.
However, this parameterization produces a relatively low number of equivalence classes using the data from our study, making for a poor comparison.
As such, we worked with the authors of PhysCov to develop a more precise abstraction, Ψ<sup>*</sup><sub>10</sub>, that yields a comparable number of equivalence classes to S<sup>3</sup>C under the *ER* and *ELR* abstractions. 
The parameterization used can be found in the [`RRS_distributions.py`](./RRS_distributions.py) file in this folder, which replaces the same folder in the original PhysCov repository.
This parameterization increases the precision of the geometric vectorization by increasing the number discrete points considered.

The RSS vectors acquired from PhysCov are available in the `s3c/study_data/physcov/` folder. The `rss_v2_dict.json` file corresponds to the new parameterization Ψ<sup>*</sup>. The JSON files contain dictionaries where the keys are 1-10 for Ψ<sub>1</sub>-Ψ<sub>10</sub>, though only Ψ<sub>10</sub> is utilized by the pipeline.