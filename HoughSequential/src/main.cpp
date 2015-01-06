#include <vector>
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <CImg.h>
#include "houghTransform.h"
#include "houghHelpers.hpp"

using namespace cimg_library;

// main method
int main(int argc, char **argv) {
	// define minimal and maximal color value for the saved images
	unsigned char minColor = 0;
	unsigned char maxColor = 255;

	// define some other variables
	double thresholdDivisor = 4;
	int excludeRadius = 20;
	int linesToExtract = 12;

	std::string filename;
	std::string resultPath = "./";
	char option;
	while ((option = getopt(argc, argv, "t:e:l:o:")) != -1) {
		switch (option) {
			case 't':
				thresholdDivisor = std::atof(optarg);
				break;
			case 'e':
				excludeRadius = std::atoi(optarg);
				break;
			case 'l':
				linesToExtract = std::atoi(optarg);
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

	// load image from file and convert to gray value
	CImg<double> inputImage(filename.c_str());
	inputImage = RGBToGrayValueImage<double>(inputImage);
	hough::HoughParameterSet<double> p(inputImage.width(), inputImage.height());

	// compute the binary image in the preprocess()-method and measure time
	std::cout << "Preprocessing..." << std::flush;
	clock_t begin = std::clock();
	CImg<bool> binaryImg = hough::preprocess<double>(inputImage, thresholdDivisor);
	clock_t end = std::clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	// compute the accumulator array and measure time
	std::cout << "Calculating accumulator array..." << std::flush;
	begin = std::clock();
	CImg<long> accumulatorArray = hough::transform<long, double>(binaryImg, p);
	end = std::clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	// compute the k best lines and measure time
	std::cout << "Extracting strongest lines..." << std::flush;
	begin = std::clock();
	std::vector<std::pair<double, double> > best = hough::extractStrongestLines<long, double>(accumulatorArray, p,
			linesToExtract, excludeRadius);
	end = std::clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	// draw the lines and measure time
	CImg<unsigned char> bestLinesImg = binaryToColorImg<unsigned char>(binaryImg);
	unsigned char redColor[] = { 255, 0, 0 };
	drawLines<unsigned char>(bestLinesImg, best, redColor);

	// display binary image, accumulator array, and strongest lines
	CImgDisplay binaryImgDisp(binaryImg, "Binary Image");
	binaryImgDisp.move(50, 50);
	CImgDisplay accDisplay(accumulatorArray, "Accumulator Array", 1);
	accDisplay.move(400, 50);
	CImgDisplay bestLinesDisp(bestLinesImg, "Best lines", 1);
	bestLinesDisp.move(0, 0);


	// save binary image, accumulator array and strongest lines image as PNG-file
	(CImg<unsigned char>(binaryImg)).normalize(minColor, maxColor).save_png(
			std::string("binaryimg.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(accumulatorArray)).normalize(minColor, maxColor).save_png(
			std::string("accumulatorarray.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(bestLinesImg)).normalize(minColor, maxColor).save_png(
			std::string("bestlines.png").insert(0, resultPath).c_str(), 3);


	// Wait until display is closed
	while (!binaryImgDisp._is_closed)
		binaryImgDisp.wait();
}

