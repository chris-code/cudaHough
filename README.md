cudaHough
========================

Repo for our parallel computing project. We aim to realize a GPU-Accellerated
version of the Hough-Transform by using the *CUDA*-environment with *C++*. To handle
load/store and displaying of images, the *CImg* library is used.
Currently, a version for detecting straight lines is implemented in both a sequential
and a parallel version.

Technical details
------------------------
Images on the GPU are represented as flat (1D) arrays in **column-major format**.

The width and height of the accumulator array represent the Theta-dimension, and
R-dimension, respectively (Where theta is the angle of the perpendicular through the
origin and R is the distance to the origin)
The dimensions of the accumulator array are calculated according to:

>dimTheta = (maxTheta - minTheta) * stepsPerRadian + 1
>dimR = (maxR - minR) * stepsPerPixel + 1;

where *stepsPerRadian* and *stepsPerPixel* determine how well the transformation will
preserve angle and distance, respektively. 1 is added because
*(maxTheta - minTehta) * stepsPerRadian* would cover the interval *[minTheta,maxTheta)*,
and we want *maxTheta* to be a valid angle.
