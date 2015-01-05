#include <vector>
#include <string>
#include <ctime>
#include <unistd.h>
#include <CImg.h>
#include "houghTransform.h"

using namespace cimg_library;

template<typename T>
CImg<T> RGBToGrayValueImage(const CImg<T> &image) {
	// initialize the gray value image
	CImg<T> grayImg(image.width(), image.height(), 1, 1);

	// iterate over the image
	for (long i = 0; i < image.width(); ++i)
		for (long j = 0; j < image.height(); ++j) {
			// The gray value is calculated by the following formula: 0.21 R + 0.72 G + 0.07 B
			grayImg(i, j, 0, 0) = 0.21 * image(i, j, 0, 0) + 0.72 * image(i, j, 0, 1) + 0.07 * image(i, j, 0, 2);
		}

	return grayImg;
}

// create an unsigned char color image from a binary image
CImg<unsigned char> binaryToColorImg(const CImg<bool> &binaryImg) {
	// initialize the color image with the color black
	CImg<unsigned char> colorImg(binaryImg.width(), binaryImg.height(), 1, 3, 0);

	// iterate over the binary image
	for (int x = 0; x < binaryImg.width(); ++x) {
		for (int y = 0; y < binaryImg.height(); ++y) {
			// if there is a 1 in the binary image, set all three RGB-values to 255
			if (binaryImg(x, y, 0, 0)) {
				for (int k = 0; k < 3; ++k)
					colorImg(x, y, 0, k) = 255;
			}
		}
	}

	// return the color image
	return colorImg;
}

// this methods draws one line in Theta-r-form with the given color in the passed image
void drawLine(CImg<unsigned char> &image, const double theta, const double r, double *color) {
	// It's a rather horizontal line (cos(Theta) could be 0)
	if ((theta >= cimg::PI / 4 && theta <= cimg::PI * 3 / 4)
			|| (theta >= cimg::PI * 5 / 4 && theta <= cimg::PI * 7 / 4)) {
		// iterate horizontally over the image
		for (int x = 0; x < image.width(); ++x) {
			// calculate y from r, Theta and x
			int y = round((r - x * cos(theta)) / sin(theta));

			// if y is within the range of the image, color the pixel (x,y)
			if (y >= 0 && y < image.height()) {
				image(x, y, 0, 0) = color[0];
				image(x, y, 0, 1) = color[1];
				image(x, y, 0, 2) = color[2];
			}
		}
	}
	// it's a rather vertical line (sin(Theta) could be 0)
	else {
		// iterate vertically over the image
		for (int y = 0; y < image.height(); ++y) {
			// calculate x from r, Theta and y
			int x = round((r - y * sin(theta)) / cos(theta));

			// if x is within the range of the image, color the pixel(x, y)
			if (x >= 0 && x < image.width()) {
				image(x, y, 0, 0) = color[0];
				image(x, y, 0, 1) = color[1];
				image(x, y, 0, 2) = color[2];
			}
		}
	}
}

// this method receives some lines in Theta-r-form and draws them in the passed image
void drawLines(CImg<unsigned char> &image, std::vector<std::pair<double, double> > &lines, double *color) {
	// iterate over lines
	for (int i = 0; i < int(lines.size()); i++) {
		// draw line
		drawLine(image, lines[i].first, lines[i].second, color);
	}
}

int main(int argc, char **argv) {
	std::string filename;
	std::string resultPath = "./";
	double threshold = 0;
	long excludeRadius = 5;
	long linesToExtract = 10;
	char option;
	while ((option = getopt(argc, argv, "t:e:l:o:")) != -1) {
		switch (option) {
			case 't':
				threshold = std::atof(optarg);
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
				if(optopt == 't' || optopt == 'e' || optopt == 'l' || optopt == 'o') {
					std::cerr << "Option -%" << optopt << " requires an argument." << std::endl;
				}
				else {
					std::cerr << "Unknown option " << optopt << std::endl;
				}
				exit(EXIT_FAILURE);
				break;
			default:
				exit(EXIT_FAILURE);
		}
	}
	if(optind >= argc) {
		std::cerr << "Path to image required" << std::endl;
		exit(EXIT_FAILURE);
	}
	filename = argv[optind];

	CImg<double> inputImage(filename.c_str()); // Load image
	inputImage = RGBToGrayValueImage<double>(inputImage);
	cudaHough::HoughParameterSet<double> hps(inputImage.width(), inputImage.height());

	std::cout << "Preprocessing..." << std::flush;
	clock_t begin = clock();
	bool *binaryImage = cudaHough::preprocess<double>(inputImage, threshold); // Transform to a binary image
	clock_t end = clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	std::cout << "Calculating accumulator array..." << std::flush;
	begin = clock();
	long *accumulatorArray = cudaHough::transform<long, double>(binaryImage, inputImage.width(), inputImage.height(),
			hps); // Transform to Hough-space
	end = clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	std::cout << "Extracting strongest lines..." << std::flush;
	begin = clock();
	std::vector<std::pair<double, double> > strongestLines = cudaHough::extractStrongestLines<long, double>(
			accumulatorArray, linesToExtract, excludeRadius, hps);
	end = clock();
	std::cout << " (" << double(end - begin) / CLOCKS_PER_SEC << "s)" << std::endl;

	CImg<bool> cpuBinaryImage = gpuToCImg<bool>(binaryImage, inputImage.width(), inputImage.height());
	CImg<long> cpuAccumulatorArray = gpuToCImg<long>(accumulatorArray, hps.getDimTheta(), hps.getDimR());

	double redColor[] = { 255, 0, 0 };
	CImg<unsigned char> cpuBestLinesImg = binaryToColorImg(cpuBinaryImage);
	drawLines(cpuBestLinesImg, strongestLines, redColor);

	CImgDisplay inputDisplay(inputImage, "Input", 1);
	CImgDisplay binaryDisplay(cpuBinaryImage, "Binary", 1);
	CImgDisplay accumulatorDisplay(cpuAccumulatorArray, "Accumulator", 1);
	CImgDisplay strongestLinesDisplay(cpuBestLinesImg, "Best Lines", 1);

	(CImg<unsigned char>(cpuBinaryImage)).normalize(0, 255).save_png(
			std::string("binaryImage.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(cpuAccumulatorArray)).normalize(0, 255).save_png(
			std::string("accumulatorArray.png").insert(0, resultPath).c_str(), 1);
	(CImg<unsigned char>(cpuBestLinesImg)).normalize(0, 255).save_png(
			std::string("bestLines.png").insert(0, resultPath).c_str(), 3);

	while (!inputDisplay.is_closed())
		inputDisplay.wait();

//	FIXME free binaryImage, accumulatorArray
	return EXIT_SUCCESS;
}
