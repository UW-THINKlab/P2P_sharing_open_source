# Peer-to-peer community sharing
 
## Requirement
- Python 3.x.
- Common libraries such as numpy, pandas, matplotlib, scipy, etc.
- folium (for visualization)

## File organization
- `RawData`: The row data used in this project.
    - `address`: The address data of the Laurelhurst and Southpark communities.
    - `survey`: The survey data of the Laurelhurst and Southpark communities.
- `data`: The derived data used in this project.
    - `preprocessing`: The processed data of the Laurelhurst and Southpark communities.
- `calibration`: The calibration results.
- `src`: The source code of this project.
    - `network.py`: The code for general networks.
    - `social_network.py`: The code to generate the social network.
    - `resource_sharing_network.py`: The code to generate the resource-sharing network.
- `test`: The test code of this project.
- `scenarios`: The scenario analysis results.
- `figs`: The figures generated in this project.

## Framework
The framework of this repo is shown as below. The modules and workflows are shown in the following sections.

![Alt text](figs/methodology_framework.png)

### Preprocessing: Process the community address data and community survey data.
- The households in the Laurelhurst community.
<img src="data/laurelhurst_address.png" width="70%">

- The households in the Southpark community.
<img src="data/southpark_address.png" width="70%">

### Coomunity-based social network

- Calibrate degree distribution. The details are shown in [laurelhurst_degree_distribution.ipynb](laurelhurst_degree_distribution.ipynb). The calibrated degree distribution is shown as below.
![Alt text](figs/laurelhurst_power_law_fit.png)
- For the southpark community, the degree distribution is shown as below.
![Alt text](figs/southpark_power_law_fit.png)

- Calibrate distance decay function. The details are shown in [distance_decay_function.ipynb](distance_decay_function.ipynb). The calibrated distance decay function is shown as below.
<img src="figs/laurelhurst_distance_decay_function.png"  width="50%">

- The final product looks like:
<img src="figs/laurelhurst_social_tie_net.png"  width="70%">
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