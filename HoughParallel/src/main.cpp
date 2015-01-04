#include <vector>
#include <string>
#include <CImg.h>
#include "houghTransform.h"

using namespace cimg_library;

void getOpts(int argc, char **argv, std::string &imgPath, std::string &resultPath, double &threshold,
		long &excludeRadius, long &linesToExtract) {
	if (argc >= 2) {
		imgPath = argv[1];
	} else {
		std::cout << "Usage: " << argv[0] << "pathToImage [resultFolder] [threshold] [excludeRadius] [lines]"
				<< std::endl;
		exit(EXIT_FAILURE);
	}

	if (argc >= 3) {
		resultPath = argv[2];
	}
	if (argc >= 4) {
		threshold = std::atof(argv[3]);
	}
	if (argc >= 5) {
		excludeRadius = std::atoi(argv[4]);
	}
	if (argc >= 6) {
		linesToExtract = std::atoi(argv[5]);
	}
}

int main(int argc, char **argv) {
	std::string filename;
	std::string resultPath = "./";
	double threshold = 0;
	long excludeRadius = 5;
	long linesToExtract = 10;
	getOpts(argc, argv, filename, resultPath, threshold, excludeRadius, linesToExtract);

	CImg<double> inputImage(filename.c_str()); // Load image
	cudaHough::HoughParameterSet<double> hps(inputImage.width(), inputImage.height());
	bool *binaryImage = cudaHough::preprocess<double>(inputImage, threshold); // Transform to a binary image
	long *accumulatorArray = cudaHough::transform<long, double>(binaryImage, inputImage.width(), inputImage.height(),
			hps); // Transform to Hough-space
//	std::vector<std::pair<double, double> > best = cudaHough::extractMostLikelyLines<double, long>(accumulatorArray,
//			linesToExtract);
	CImg<bool> cpuBinaryImage = gpuToCImg<bool>(binaryImage, inputImage.width(), inputImage.height());
	CImg<long> cpuAccumulatorArray = gpuToCImg<long>(accumulatorArray, hps.getDimTheta(), hps.getDimR());

	CImgDisplay inputDisplay(inputImage, "Input", 1);
	CImgDisplay binaryDisplay(cpuBinaryImage, "Binary", 1);
	CImgDisplay accumulatorDisplay(cpuAccumulatorArray, "Accumulator", 1);

	cpuBinaryImage.normalize(0, 255);
	cpuAccumulatorArray.normalize(0, 255);
	cpuBinaryImage.save_png(std::string("binaryImage.png").insert(0, resultPath).c_str(), 1);
	cpuAccumulatorArray.save_png(std::string("accumulatorArray.png").insert(0, resultPath).c_str(), 1);

	while (!inputDisplay.is_closed())
		inputDisplay.wait();

	return EXIT_SUCCESS;
}
