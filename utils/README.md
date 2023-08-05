# Utilities for working with ASGs
This folder contains utility functions for working with ASGs and their clusterings.
The folder is organized as follows:
* `test/`
  * Python unit tests for the utility functions
* `asg_compare.py`
  * Functions for comparing ASGs, including the intermediate graph hashing described in Section 3.2.3. Note that the `no-id` abstraction described in the paper is implemented directly in the ASG comparison metrics - for bookkeeping, the IDs of entities are kept in the graph until the isomorphism check, at which point they are ignored.
* `cluster_generator.py`
  * Command Line options for generating the clusterings.
* `dataset.py`
  * The `Dataset` class functions as the set of clustered ASGs and calling the constructor will generate the clustering. Also contains several helper methods for working with clusters, and can be serialized to a JSON file, e.g., the ones found in `study_data/carla_clusters/`.
* `empty_sg.pkl`
  * As described in Section 3.2.3, special handling for the empty SG can speed up computation. This pickle contains an empty SG for reference.