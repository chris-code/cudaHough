#include <vector>
#include <string>
#include <CImg.h>
#include "houghTransform.h"

using namespace cimg_library;

int main(int argc, char **argv) {
	std::string filename = "images/stoppschild3.jpg";
	if (argc >= 2) {
		filename = argv[1];
	}
	short linesToExtract = 10;
	if (argc >= 3) {
		linesToExtract = std::atoi(argv[2]);
	}

	CImg<double> inputImage(filename.c_str()); // Load image
	CImg<bool> binaryImage = cudaHough::preprocess(inputImage); // Transform to a binary image
	CImg<long> accumulatorArray = cudaHough::transform(binaryImage); // Transform to Hough-space
	std::vector< std::pair<double, double> > best = cudaHough::extractMostLikelyLines(accumulatorArray, linesToExtract);
	return EXIT_SUCCESS;
}
