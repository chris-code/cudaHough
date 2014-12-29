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
	cudaHough::HoughParameterSet<double> hps(inputImage.width(), inputImage.height());
	bool *binaryImage = cudaHough::preprocess<double>(inputImage, threshold); // Transform to a binary image
	long *accumulatorArray = cudaHough::transform<long, double>(binaryImage, inputImage.width(), inputImage.height(),
			hps); // Transform to Hough-space
//	std::vector<std::pair<double, double> > best = cudaHough::extractMostLikelyLines<double, long>(accumulatorArray,
//			linesToExtract);

	CImgDisplay inputDisplay(inputImage, "Input", 1);
	CImgDisplay binaryDisplay(gpuToCImg<bool>(binaryImage, inputImage.width(), inputImage.height()), "Binary", 1);
	CImgDisplay accumulatorDisplay(gpuToCImg<long>(accumulatorArray, inputImage.width(), inputImage.height()),
			"Accumulator", 1);

	while (!inputDisplay.is_closed())
		inputDisplay.wait();

	return EXIT_SUCCESS;
}
