TTNN: A Physically-Guided Deep Learning Model for Focal Depth and Epicenter Distance Estimation Based on Multistation Waveforms
=
The codes show how to use the TTNN for estimating the focal depth and epicenter distance.

The codes are developed based on Python with Pytorch and the description of the codes are as follows:

./TTNN/model.py: This code is for generating the TTNN model for estimation the focal depth and epicenter distance. A function named predictor is provided to invoke the model and generate outputs.

./TTNN/evaluation.py: This code provides two main functions, evaluate and calculate_epicenter. The evaluate function is used to assess TTNN's estimation performance for the focal depth and epicentral distance, using four metrics: mean error, mean absolute error, error standard deviation, and R². The calculate_epicenter function calculates the latitude and longitude of the epicenter using the grid search method.

./usage.ipynb: This file provides an example of how to use TTNN.
