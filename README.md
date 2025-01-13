This repository contains the code for the paper "Homophily and geography in online follower ties". All files are Jupyter Notebooks, except to python scripts with functions that are used in the notebooks. 

- `Build_network.ipynb` contains the code to construct the follower network from the raw follower data, match it to the voter file and to the other sources of node and edge attributes (census tract RUCA and population density and number of election tweets), pre-process the attributes, calculate the distance between each edge, and add population in the radius from the `Radiation_model.ipynb` notebook. The output is a pickle file, `follower_graph.pk`, which is the pickled version of a NetworkX object, the input of all the other notebooks. This code also outputs the composition of the Twitter Panel for Appendix Table 1. 
- `Descriptives.ipynb` outputs the network composition and average degrees for Appendix Table 1.  
- `Radiation_model.ipynb` calculates the population in the radius between each edge in the network, and between random samples of dyads. The values between random samples of dyads are saved as `radiation_pop_rand_labels.pk` and `radiation_pop_rand_labels_2.pk` files.
- `Network_Composition.ipynb` calculates and plots the results for Measure 1. 
-  `Bivariate_analysis.ipynb` contains the code for the plots of Measures 3, and all the results of Measure 2 with categorical variables. 
-  `two_variable_heatmaps.ipynb` plots the trivariate heatmaps of Appendix Sections 7 and 10
-  `homophily_functions.py` has functions doing the bulk of the calculations in `Network_Composition.ipynb`, `Bivariate_analysis.ipynb`, and `two_variable_heatmaps.ipynb`.
-  `Logistic.ipynb` and `logistic_func.py` contain the code for the logistic regressions of Measure two with age, distance, and population in radius, and for the multivariate logistic regressions and the KHB method.
-  `CSP_data.ipynb` pre-processes the Covid States survey data and network, and matches with the voter file data to run the comparison of Appendix Section 3. `CSP_homophily.ipynb` runs the homophily results with the Covid States data of Appendix Section 4. 
