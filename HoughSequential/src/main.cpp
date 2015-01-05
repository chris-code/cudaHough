#include <CImg.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <utility>
#include <ctime>
#include "houghTransform.h"
#include "houghHelpers.hpp"

using namespace cimg_library;


// main method
int main(int argc, char **argv)
{
	// define minimal and maximal color value for the saved images
	unsigned char minColor = 0;
	unsigned char maxColor = 255;

	// define some other variables
	double thresholdDivisor = 4;
	int excludeRadius = 20;
	int linesToExtract = 12;

	std::string filename;
	std::string resultPath = "./";
	if(argc >= 2)
		filename = argv[1];
	if (argc >= 3)
		resultPath = argv[2];
	if (argc >= 4)
		thresholdDivisor = atoi(argv[3]);
	if (argc >= 5)
		excludeRadius = atoi(argv[4]);
	if (argc >= 6)
		linesToExtract = atoi(argv[5]);

	// load image from filename
	CImg<double> img(filename.c_str());

	// convert it to a grayvalue image
	CImg<double> grayImg = RGBToGrayValueImage<double>(img);

	// compute the binary image in the preprocess()-method and measure time
	clock_t preprocessStart = std::clock();
	CImg<bool> binaryImg = hough::preprocess(grayImg, thresholdDivisor);
	clock_t preprocessEnd = std::clock();

	// print how much time it took to compute the binary image
	std::cout << "Preprocess time: " << double(preprocessEnd - preprocessStart) / CLOCKS_PER_SEC << std::endl;

	// display the binary image
	CImgDisplay binaryImgDisp(binaryImg, "Binary Image");
	binaryImgDisp.move(50, 50);

	// save binary image as PNG-file
	(CImg<unsigned char> (binaryImg)).normalize(minColor, maxColor).save_png(std::string("binaryimg.png").insert(0, resultPath).c_str(), 1);

	hough::HoughParameterSet<double> p(binaryImg.width(), binaryImg.height());

	// compute the accumulator array and measure time
	clock_t houghStart = std::clock();
	CImg<long> accumulatorArray = hough::transform(binaryImg, p);
	clock_t houghEnd = std::clock();

	// print how much time it took to compute the accumulator array
	std::cout << "Hough time: " << double(houghEnd - houghStart) / CLOCKS_PER_SEC << std::endl;

	// save accumulator array as PNG-file
	(CImg<unsigned char> (accumulatorArray)).normalize(minColor, maxColor).save_png(std::string("accumulatorarray.png").insert(0, resultPath).c_str(), 1);

	// display the accumulator array
	CImgDisplay accDisplay(accumulatorArray, "Accumulator Array", 1);
	accDisplay.move(400,50);

	// compute the k best lines and measure time
	clock_t bestStart = std::clock();
	std::vector< std::pair<double, double> > best = hough::extractStrongestLines(accumulatorArray, p, linesToExtract,
			excludeRadius);
	clock_t bestEnd = std::clock();

	// print how much time it took to compute the k best lines
	std::cout << "Best lines time: " << double(bestEnd - bestStart) / CLOCKS_PER_SEC << std::endl;

	// draw the lines and measure time
	CImg<unsigned char> bestLinesImg = binaryToColorImg<unsigned char>(binaryImg);
	unsigned char redColor[] = {255, 0, 0};
	clock_t drawStart = std::clock();
	drawLines<unsigned char>(bestLinesImg, best, redColor);
	clock_t drawEnd = std::clock();

	// print how long it took to draw the k best lines
	std::cout << "Draw time: " << double(drawEnd - drawStart) / CLOCKS_PER_SEC << std::endl;

	// save best line image as PNG-file
	(CImg<unsigned char> (bestLinesImg)).normalize(minColor, maxColor).save_png(std::string("bestlines.png").insert(0, resultPath).c_str(), 3);

	// display the best lines image
	CImgDisplay bestLinesDisp(bestLinesImg, "Best lines", 1);
	bestLinesDisp.move(0, 0);

	// Wait until display is closed
	while (!binaryImgDisp._is_closed)
		binaryImgDisp.wait();
}

