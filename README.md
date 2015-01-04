cudaHough
========================

Repo for our parallel computing project. We aim to realize a GPU-Accellerated
version of the Hough-Transform by using the CUDA-environment.


The width of the accumulator array represents the Theta-dimension,
the height of the accumulator array respectivly the R-dimension

The dimensions of Theta and R are calculated by the following formulas:

dimTheta = (maxTheta - minTheta) * stepsPerRadian + 1
dimR = (maxR - minR) * stepsPerPixel + 1;

We add 1, because we (maxTheta - minTehta) * stepsPerRadian would cover the
interval [minTheta,maxTheta) and we need one more slot for maxTheta.
