# Peer-to-peer resource sharing

Code for paper "Untapped Capacity of Place-based Peer-to-Peer (P2P) Resource Sharing for Community Resilience" by Zhengyang Li, Katherine Idziorek, Anthony Chen, Cynthia Chen.
 
## Requirements
- Python 3.x.
- Common libraries such as numpy, pandas, matplotlib, scipy, etc.
- Gurobi 10.0.2 (for optimization)
- Networkx 3.1 (for network analysis)

## Installation guide
Clone this repo.

## Demo
See the `Demo` folder for the demo code and the expected output.

## File organization
- `data`: The derived data used in this project.
- `src`: The source code of this project.
    - `community.py`: The code for generating community-based social networks and P2P resource-sharing networks.
    - `resource_sharing_model.py`: The resource shairng model.
    - `evaluation_metrics.py`: The evaluation metrics for resilience loss.
- `results`: The scenario analysis results.
- `demo`: The demo code and expected output.
- `figs`: The figures generated in this project.

## Framework
The framework of this repo is shown as below. The modules and workflows are shown in the following sections.

![Alt text](figs/methodology_framework.png)

### Coomunity-based social network

Degree distribution.
- Laurelhurst.
<img src="figs/laurelhurst_negative_binomial_fit.png"  width="50%">
- South Park
<img src="figs/southpark_negative_binomial_fit.png"  width="50%">

Distance decay function. 
- Laurelhurst.
<img src="figs/laurelhurst_distance_decay_function.png"  width="50%">
- South Park.
<img src="figs/southpark_distance_decay_function.png"  width="50%">

An example of the generated social network.
- Laurelhurst.
<img src="figs/laurelhurst_social_tie_net.png"  width="70%">
- South Park.
<img src="figs/southpark_social_tie_net.png"  width="70%">

### Community-based resource-sharing network

- Willingness-to-share. See [willingness_to_share.ipynb](willingness_to_share.ipynb). The calibrated willingness-to-share is shown as below.
<img src="figs/laurelhurst_willingness_to_share.png"  width="50%">

- Generating social network and its attributes. [generate_social_network.py](scr/generate_social_network.py)

- Resource distribution. See [laurelhurst_resource_distribution.ipynb](laurelhurst_resource_distribution.ipynb).
<img src="figs/laurelhurst_transp_resource_dist.png"  width="50%">

- Resource distribution in the Southpark community. See [southpark_resource_distribution.ipynb](southpark_resource_distribution.ipynb).
<img src="figs/southpark_transp_resource_dist.png"  width="50%">

### Resource-sharing model
- Resource sharing model. [resource_sharing_model.py](src/resource_sharing_model.py)

- Before sharing.
<img src="figs/laurelhurst_transp_resource_inventory.png"  width="70%">

- After sharing.
<img src="figs/laurelhurst_transp_resource_redistribution.png"  width="70%">

### Evaluation matrics
- See [evaluation_metrics.py](src/evaluation_metrics.py)