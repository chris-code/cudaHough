#include <vector>
#include <ctime>
#include <string>
#include <iostream>
#include <unistd.h>
#include <CImg.h>
#include "houghTransform.h"
#include "houghHelpers.hpp"

using namespace cimg_library;

template<typename imgT, typename accuT, typename paramT>
void execute(std::string &filename, std::string &resultPath, double threshold, long excludeRadius,
		long linesToExtract) {
	// load image from file and convert to gray value
	CImg<imgT> inputImage(filename.c_str());
	inputImage = RGBToGrayValueImage<imgT>(inputImage);
	hough::HoughParameterSet<paramT> p(inputImage.width(), inputImage.height());

	// compute the binary image in the preprocess()-method and measure time
	std::cout << "Preprocessing..." << std::flush;
	clock_t begin = std::clock();
	CImg<bool> binaryImg = hough::preprocess<imgT>(inputImage, threshold);
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
	unsigned char redColor[] = { 255, 0, 0 };
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

int main(int argc, char **argv) {
	std::string filename;
	std::string resultPath = "./";
	double threshold = 0.5;
	long excludeRadius = 20;
	long linesToExtract = 16;

	char option;
	while ((option = getopt(argc, argv, "t:e:l:o:")) != -1) {
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

	execute<double, long, double>(filename, resultPath, threshold, excludeRadius, linesToExtract);
	return EXIT_SUCCESS;
}
