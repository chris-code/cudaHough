#include <vector>
#include <ctime>
#include <string>
#include <iostream>
#include <getopt.h>
#include <CImg.h>
#include "houghSequential.h"
#include "houghParallel.h"
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

template<typename imgT, typename accuT, typename paramT>
void executeSequential(std::string &filename, std::string &resultPath, double threshold, double sigma,
	long excludeRadius,
	long linesToExtract) {
	// load image from file and convert to gray value
	CImg<imgT> inputImage(filename.c_str());
	inputImage = RGBToGrayValueImage<imgT>(inputImage);
	hough::HoughParameterSet<paramT> p(inputImage.width(), inputImage.height());

	// compute the binary image in the preprocess()-method and measure time
	std::cout << "Preprocessing..." << std::flush;
	clock_t begin = std::clock();
	CImg<bool> binaryImg = hough::preprocess<imgT>(inputImage, threshold, sigma);
	clock_t end = std::clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	// compute the accumulator array and measure time
	std::cout << "Calculating accumulator array..." << std::flush;
	begin = std::clock();
	CImg<accuT> accumulatorArray = hough::transform<accuT, paramT>(binaryImg, p);
	end = std::clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	// compute the k best lines and measure time
	std::cout << "Extracting strongest lines..." << std::flush;
	begin = std::clock();
	std::vector<std::pair<paramT, paramT> > best = hough::extractStrongestLines<accuT, paramT>(accumulatorArray, p,
		linesToExtract, excludeRadius);
	end = std::clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	// draw the lines and measure time
	CImg<unsigned char> bestLinesImg = binaryToColorImg<unsigned char>(binaryImg);
	unsigned char redColor[] = {255, 0, 0};
	drawLines<unsigned char>(bestLinesImg, best, redColor);

	// display binary image and strongest lines
	CImgDisplay binaryDisplay(binaryImg, "Binary Image");
	CImgDisplay bestLinesDisplay(bestLinesImg, "Best lines", 1);

	// save binary image, accumulator array and strongest lines image as PNG-file
	unsigned char minColor = 0;
	unsigned char maxColor = 255;
	(CImg<unsigned char>(binaryImg)).normalize(minColor, maxColor).save_png(
		std::string("binaryimg.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(accumulatorArray)).normalize(minColor, maxColor).save_png(
		std::string("accumulatorarray.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(bestLinesImg)).normalize(minColor, maxColor).save_png(
		std::string("bestlines.png").insert(0, resultPath).c_str(), 3);

	// Wait until display is closed
	while (!binaryDisplay._is_closed)
		binaryDisplay.wait();
}

//	Main is responsible for processing command line parameters
int main(int argc, char **argv) {
	bool useCuda = false;
	bool useDoublePrecision = false;
	std::string filename;
	std::string resultPath = "./";
	double threshold = 0.5;
	double sigma = 2.0;
	long excludeRadius = 20;
	long linesToExtract = 16;

	struct option options[] = {
		{"cuda", no_argument, NULL, 'c'},
		{"double", no_argument, NULL, 'd'},
		{"threshold", required_argument, NULL, 't'},
		{"sigma", required_argument, NULL, 's'},
		{"exclude-radius", required_argument, NULL, 'e'},
		{"lines", required_argument, NULL, 'l'},
		{"output-path", required_argument, NULL, 'o'},
		{0, 0, NULL, 0}};
	char option;
	while ((option = getopt_long(argc, argv, "cdt:s:e:l:o:", options, NULL)) != -1) {
		switch (option) {
			case 'c':
				useCuda = true;
				break;
			case 'd':
				useDoublePrecision = true;
				break;
			case 't':
				threshold = std::atof(optarg);
				break;
			case 's':
				sigma = std::atof(optarg);
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
				if (optopt == 't' || optopt == 's' || optopt == 'e' || optopt == 'l' || optopt == 'o') {
					std::cerr << "Option -%" << char(optopt) << " requires an argument." << std::endl;
				} else {
					std::cerr << "Unknown option " << char(optopt) << std::endl;
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

	if (useCuda && !useDoublePrecision) {
		execute<float, long, float>(filename, resultPath, threshold, sigma, excludeRadius, linesToExtract);
	}
	else if (useCuda && useDoublePrecision) {
		execute<double, long, double>(filename, resultPath, threshold, sigma, excludeRadius, linesToExtract);
	}
	else if (!useCuda && !useDoublePrecision) {
		executeSequential<float, long, float>(filename, resultPath, threshold, sigma, excludeRadius, linesToExtract);
	}
	else if (!useCuda && useDoublePrecision) {
		executeSequential<double, long, double>(filename, resultPath, threshold, sigma, excludeRadius, linesToExtract);
	}
	return EXIT_SUCCESS;
}
