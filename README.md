cudaHough
========================

The purpose of this project is to realize a GPU-Accellerated version of the Hough-
Transform by using the *CUDA*-environment with *C++*. To handle load/store and
displaying of images, the *CImg* library is used. Currently, a version for detecting
straight lines is implemented as both a sequential and a parallel version.

Technical details
------------------------
Images on the GPU are represented as flat (1D) arrays in **column-major format**.

Throughout the parallel version, templates are used. This can get quite messy in C++,
so here are our naming conventions for template type names:
- paramT - Type for hough transform parameters. Usually floats of varying precision
- accuT - The type of the accumulator array. Usually an integer.
- imgT - Type parameter for the image (usually a CImg object)

The width and height of the accumulator array represent the Theta-dimension and
R-dimension, respectively (where Theta is the angle of the perpendicular through the
origin and R is the distance to the origin)
The dimensions of the accumulator array are calculated according to:

>dimTheta = (maxTheta - minTheta) * stepsPerRadian + 1
>dimR = (maxR - minR) * stepsPerPixel + 1;

where *stepsPerRadian* and *stepsPerPixel* determine how well the transformation will
preserve angle and distance, respectively. 1 is added because
*(maxTheta - minTehta) * stepsPerRadian* would cover the interval *[minTheta,maxTheta)*,
and we want *maxTheta* to be a valid angle.
