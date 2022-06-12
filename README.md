# MSTS
This Repository consists of code to run a correlation based feature subset selection technique for multivariate time series data called MSTS

### Data
The code runs the data from the UEA multivariate time series archive and comapres against a wrapper approach

There are 4 main functions presented here:

DTW_MSTS - Runs the MSTS technique using a 1NN-DTW classifier //
Rocket_MSTS - Runs the MSTS technique using a Rocket classifier 
DTW_knn - Runs the knn based MI technique using a DTW classifier 
Rocket_knn - Runs the knn based MI technique using a Rocket classifier

When using this code, kindly cite the following paper:

Kathirgamanathan, B., Cunningham, P.: Correlation based feature subset selection for multivariate time-series data. arXiv preprint arXiv:2112.03705 (2021) https://arxiv.org/pdf/2112.03705.pdf
