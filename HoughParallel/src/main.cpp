#include <vector>
#include <string>
#include <CImg.h>
#include "houghTransform.h"

using namespace cimg_library;

int main(int argc, char **argv) {
	std::string filename = "images/stoppschild3.jpg";
	double threshold = 0;
	short linesToExtract = 10;
	if (argc >= 2) {
		filename = argv[1];
	}
	if (argc >= 3) {
		threshold = std::atof(argv[2]);
	}
	if (argc >= 4) {
		linesToExtract = std::atoi(argv[3]);
	}
	CImg<double> inputImage(filename.c_str()); // Load image
	bool *binaryImage = cudaHough::preprocess<double>(inputImage, threshold); // Transform to a binary image
	CImg<long> accumulatorArray = cudaHough::transform(binaryImage); // Transform to Hough-space
	std::vector< std::pair<double, double> > best = cudaHough::extractMostLikelyLines(accumulatorArray, linesToExtract);
	return EXIT_SUCCESS;
}
