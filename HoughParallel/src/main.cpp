#include <vector>
#include <ctime>
#include <string>
#include <iostream>
#include <getopt.h>
#include <CImg.h>
#include "houghTransform.h"
#include "houghHelpers.hpp"

using namespace cimg_library;

template<typename imgT, typename accuT, typename paramT> //TODO use template parameters
void execute(std::string &filename, std::string &resultPath, double threshold, double sigma, long excludeRadius,
	long linesToExtract) {
	CImg<imgT> inputImage(filename.c_str()); // Load image
	inputImage = RGBToGrayValueImage<imgT>(inputImage);
	cudaHough::HoughParameterSet<paramT> hps(inputImage.width(), inputImage.height());

	std::cout << "Preprocessing..." << std::flush;
	clock_t begin = clock();
	bool *binaryImage = cudaHough::preprocess<imgT>(inputImage, threshold, sigma); // Transform to a binary image
	clock_t end = clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	std::cout << "Calculating accumulator array..." << std::flush;
	begin = clock();
	accuT *accumulatorArray = cudaHough::transform<accuT, paramT>(binaryImage, inputImage.width(), inputImage.height(),
		hps); // Transform to Hough-space
	end = clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	std::cout << "Extracting strongest lines..." << std::flush;
	begin = clock();
	std::vector<std::pair<paramT, paramT> > strongestLines = cudaHough::extractStrongestLines<accuT, paramT>(
		accumulatorArray, linesToExtract, excludeRadius, hps);
	end = clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	CImg<bool> cpuBinaryImage = gpuToCImg<bool>(binaryImage, inputImage.width(), inputImage.height());
	CImg<accuT> cpuAccumulatorArray = gpuToCImg<accuT>(accumulatorArray, hps.getDimTheta(), hps.getDimR());

	unsigned char redColor[] = {255, 0, 0};
	CImg<unsigned char> cpuBestLinesImg = binaryToColorImg<unsigned char>(cpuBinaryImage);
	drawLines<unsigned char>(cpuBestLinesImg, strongestLines, redColor);

	CImgDisplay binaryDisplay(cpuBinaryImage, "Binary", 1);
	CImgDisplay strongestLinesDisplay(cpuBestLinesImg, "Best Lines", 1);

	(CImg<unsigned char>(cpuBinaryImage)).normalize(0, 255).save_png(
		std::string("binaryImage.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(cpuAccumulatorArray)).normalize(0, 255).save_png(
		std::string("accumulatorArray.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(cpuBestLinesImg)).normalize(0, 255).save_png(
		std::string("bestLines.png").insert(0, resultPath).c_str(), 3);

	while (!binaryDisplay.is_closed())
		binaryDisplay.wait();
}

int main(int argc, char **argv) {
	std::string filename;
	std::string resultPath = "./";
	double threshold = 0.5;
	long excludeRadius = 20;
	long linesToExtract = 16;

	struct option options[] = {
		{"threshold", required_argument, NULL, 't'},
		{"exclude-radius", required_argument, NULL, 'e'},
		{"lines", required_argument, NULL, 'l'},
		{"output-path", required_argument, NULL, 'o'},
		{0, 0, NULL, 0}};
	char option;
	while ((option = getopt_long(argc, argv, "t:e:l:o:", options, NULL)) != -1) {
		switch (option) {
			case 't':
				threshold = std::atof(optarg);
				break;
			case 'e':
				excludeRadius = std::atol(optarg);
				break;
			case 'l':
				linesToExtract = std::atol(optarg);
				break;
			case 'o':
				resultPath = optarg;
				break;
			case '?':
				if (optopt == 't' || optopt == 'e' || optopt == 'l' || optopt == 'o') {
					std::cerr << "Option -%" << optopt << " requires an argument." << std::endl;
				} else {
					std::cerr << "Unknown option " << optopt << std::endl;
				}
				exit(EXIT_FAILURE);
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}
	if (optind >= argc) {
		std::cerr << "Path to image required" << std::endl;
		exit(EXIT_FAILURE);
	}
	filename = argv[optind];

	double sigma = 2.0;

	execute<double, long, double>(filename, resultPath, threshold, sigma, excludeRadius, linesToExtract);
	return EXIT_SUCCESS;
}
