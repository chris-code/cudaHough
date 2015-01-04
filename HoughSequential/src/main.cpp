#include <CImg.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <ctime>
#include "HoughParameterSet.h"

using namespace cimg_library;


// this methods prints the color values of an image
template <typename T>
void printImg(const CImg<T>& image)
{
	// iterate over the image and print every entry to the console
	for (int i = 0; i < image.width(); i++)
	{
		for (int j = 0; j < image.height(); j++)
		{
			std::cout << image(i, j, 0, 0) << "\t";
		}
		std::cout << std::endl;
	}
}


// this method transforms a rgb-color image to an grayvalue image
CImg<double> RGBToGrayValueImage(const CImg<double>& image)
{
	// initialize the gray value image
	CImg<double> grayImg(image.width(), image.height(), 1, 1);

	// temp variable
	double grayvalue = 0;

	// iterate over the image
	for (int i = 0; i < image.width(); i++)
		for (int j = 0; j < image.height(); j++)
		{
			// The gray value is calculated by the following formula: 0.21 R + 0.72 G + 0.07 B
			grayvalue = 0.21 * image(i, j, 0, 0) + 0.72 * image(i, j, 0, 1) + 0.07 * image(i, j, 0, 2);
			grayImg(i,j,0,0) = grayvalue;
		}

	// return the gray value image
	return grayImg;
}


// this methods normalizes an image so that the sum of its pixels equals 1
CImg<double> normalize(const CImg<double>& filter)
{
	double sum = 0.;

	// get the sum of all filter values
	for (int i = 0; i < filter.width(); i++)
	{
		for (int j = 0; j < filter.height(); j++)
		{
			sum += filter(i, j, 0, 0);
		}
	}

	// initialize the normalized filter
	CImg<double> normalizedFilter(filter.width(), filter.height());

	// divide every value in the filter by the sum
	for (int i = 0; i < filter.width(); i++)
	{
		for (int j = 0; j < filter.height(); j++)
		{
			normalizedFilter(i, j, 0, 0) = filter(i, j, 0, 0) / sum;
		}
	}

	// return the normalized filter
	return normalizedFilter;
}


// this methods convolves an gray value image with a filter using the Wrap-Around approach
CImg<double> convolve(const CImg<double>& image, const CImg<double>& filter, const int offsetX, const int offsetY)
{
	// initialize the convolved image
	CImg<double> convolvedImg(image.width(), image.height(), 1, 1);

	// iterate over image
	for (int imgX = 0; imgX < image.width(); imgX++)
	{
		for (int imgY = 0; imgY < image.height(); imgY++)
		{
			// set the temporary sum to 0
			double tempSum = 0;

			// iterate over filter in x dimension
			for (int filX = 0; filX < filter.width(); filX++)
			{
				// calculate the x position in the image
				int posImgX = ((imgX - offsetX + filX) + image.width()) % image.width();

				// iterate over the filter in y-dimension
				for (int filY = 0; filY < filter.height(); filY++)
				{
					// calculate the y position in the image
					int posImgY = ((imgY - offsetY + filY) + image.height()) % image.height();

					// add the product of the values in the image and the filter to the temporary sum
					tempSum += image(posImgX, posImgY, 0, 0) * filter(filX, filY, 0, 0);
				}
			}

			// set the value at position (imgX, imgY) in the convolved image to the computed temporary sum
			convolvedImg(imgX, imgY, 0, 0) = tempSum;
		}
	}

	// return the convolved image
	return convolvedImg;
}


// this method calculates the gradient strength of an gray value image given the results of the convolution with SobelX and SobelY
CImg<double> calculateGradientStrength(const CImg<double>& sobelX, const CImg<double>& sobelY)
{
	// initialize the gradient strength image
	CImg<double> strengthImg(sobelX.width(), sobelX.height(), 1, 1);

	// iterate over the strength image
	for (int x = 0; x < sobelX.width(); x++)
		for (int y = 0; y < sobelX.height(); y++)
			// calculate the strength of the gradient at position (x,y) from the values in sobelX(x,y) and sobelY(x,y)
			strengthImg(x,y,0,0) = sqrt(sobelX(x,y,0,0) * sobelX(x,y,0,0) + sobelY(x,y,0,0) * sobelY(x,y,0,0));

	// return the strength image
	return strengthImg;
}


// returns a binary image given a grayvalue image
CImg<bool> makeBinaryImage(const CImg<double>& image, const double threshold)
{
	// initialize the binary image
	CImg<bool> binaryImg(image.width(), image.height(), 1, 1);

	// iterate over the image
	for (int x = 0; x < image.width(); x++)
		for (int y = 0; y < image.height(); y++)
			// if the value in the original image is greater than the threshold, the pixel (x,y) becomes a 1 pixel
			if (image(x,y,0,0) > threshold)
				binaryImg(x,y,0,0) = true;
			// else it becomes a 0 pixel
			else
				binaryImg(x,y,0,0) = false;

	// return the binary image
	return binaryImg;
}


// this methods converts the input image to the binary image needed by the Hough transform
CImg<bool> preprocess(const char* filename, double thresholdDivisor = 2)
{
	// load image from filename
	CImg<double> img(filename);

	// convert it to a grayvalue image
	CImg<double> grayImg = RGBToGrayValueImage(img);

	// create Sobel X filter
	double sobelXarr[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	CImg<double> sobelX(sobelXarr, 3, 3);

	// create Sobel Y filter
	double sobelYarr[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	CImg<double> sobelY(sobelYarr, 3, 3);

	// convolve Image with both Sobel filters
	CImg<double> sobelXImg = convolve(grayImg, sobelX, 1, 1);
	CImg<double> sobelYImg = convolve(grayImg, sobelY, 1, 1);

	// calculate the gradient strength
	CImg<double> strengthImg = calculateGradientStrength(sobelXImg, sobelYImg);

	// create binomial filter
	double binomialArr[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	CImg<double> binomialUnnormalized(binomialArr, 3, 3);
	CImg<double> binomial = normalize(binomialUnnormalized);

	// smooth image
	strengthImg = convolve(strengthImg, binomial, 4, 4);

	// calculate the binary image of the gradient strength image
	double threshold = (strengthImg.min() + strengthImg.max()) / thresholdDivisor;

	// make and return binary image
	return makeBinaryImage(strengthImg, threshold);
}


// this methods computes the accumulator array
CImg<long> computeAccumulatorArray(const CImg<bool>& binaryImg, const HoughParameterSet& p)
{
	// calculate the dimensions of the accumulator array by the given HoughParameterSet
	int dimTheta = (p.maxTheta - p.minTheta) * p.stepsPerRadian;
	int dimR = (p.maxR - p.minR) * p.stepsPerPixel;

	// initialize the border exclude value, which defines how much of the border is not considered
	int borderExclude = 5;

	// calculate the thetaStepsize by inverting the steps per radian
	double thetaStepsize = 1 / p.stepsPerRadian;

	// initialize the accumulator array as black image (initially every line has 0 votes)
	CImg<long> accumulatorArray(dimTheta, dimR, 1, 1, 0);

	// iterate over the image and ignore some points at the border
	for (int x = borderExclude; x < binaryImg.width() - borderExclude; x++)
	{
		for (int y = borderExclude; y < binaryImg.height() - borderExclude; y++)
		{
			// if there is written a 1 in the binary image, increment the vote matrix at the corresponding positions
			if (binaryImg(x, y, 0, 0))
			{
				// iterate over all possible values for Theta
				for (double theta = p.minTheta; theta <= p.maxTheta; theta += thetaStepsize)
				{
					// calculate the r value
					double r = x * cos(theta) + y * sin(theta);

					// calculate the index in the accumulator array
					int thetaIdx = int((theta - p.minTheta) * p.stepsPerRadian);
					int rIdx = int((r - p.minR) * p.stepsPerPixel);

					// increment the value at the calculated position
					accumulatorArray(thetaIdx, rIdx, 0, 0) = accumulatorArray(thetaIdx, rIdx, 0, 0) +  1;
				}
			}
		}
	}

	// return the accumulator array
	return accumulatorArray;
}


// create an unsigned char color image from a binary image
CImg<unsigned char> binaryToColorImg(const CImg<bool>& binaryImg)
{
	// initialize the color image with the color black
	CImg<unsigned char> colorImg(binaryImg.width(), binaryImg.height(), 1, 3, 0);

	// iterate over the binary image
	for (int x = 0; x < binaryImg.width(); x++)
	{
		for (int y = 0; y < binaryImg.height(); y++)
		{
			// if there is a 1 in the binary image, set all three RGB-values to 255
			if (binaryImg(x, y, 0, 0))
			{
				for (int k = 0; k < 3; k++)
					colorImg(x, y, 0, k) = 255;
			}
		}
	}

	// return the color image
	return colorImg;
}


// this methods finds local optima in an image
template <typename T>
std::vector< std::vector<int> > getLocalMaxima(const CImg<T>& image, int excludeRadius)
{
	// declare vector that shall save the maxima
	std::vector< std::vector<int> > maxima;

	// iterate over the image
	for (int x = 0; x < image.width(); x++)
	{
		for (int y = 0; y < image.height(); y++)
		{
			bool isMaximum = true;

			// make sure that there is no better point within a square with "radius" excludeRadius
			for (int i = -excludeRadius; i <= excludeRadius; i++)
			{
				int posX = ((x + i) + image.width()) % image.width();

				for (int j = -excludeRadius; j <= excludeRadius; j++)
				{
					int posY = ((y + j) + image.height()) % image.height();

					if (image(posX, posY, 0, 0) >= image(x, y, 0, 0) && (posX != x || posY != y))
						isMaximum = false;
				}
			}

			// if there is no better point, add the point/pixel to the maxima vector
			if (isMaximum)
			{
				std::vector<int> m;
				m.push_back(x);
				m.push_back(y);
				m.push_back(image(x,y,0,0));

				maxima.push_back(m);
			}

		}
	}

	// return the maxima vector
	return maxima;
}


// this method is a comparator for lines given by a 3-tupel r, Theta and entry of the vote matrix/accumulator array
bool compareLines(std::vector<int> v1, std::vector<int> v2)
{
	// return true, if the first vector is bigger than the second one
	return v1[2] > v2[2];
}


// this methods extracts the k best lines from the accumulator array
std::vector< std::pair<double, double> > getKBestLines(const CImg<long>& accArray, const HoughParameterSet& p, int k, int excludeRadius)
{
	clock_t findMaximaStart = std::clock();
	// compute local maxima
	std::vector< std::vector<int> > maxima = getLocalMaxima(accArray, excludeRadius);
	clock_t findMaximaEnd = std::clock();

	// print how much time it took to find the local maxima
	std::cout << "Find maxima time: " << double(findMaximaEnd - findMaximaStart) / CLOCKS_PER_SEC << std::endl;

	clock_t sortStart = std::clock();
	// sort them
	std::sort(maxima.begin(), maxima.end(), compareLines);
	clock_t sortEnd = std::clock();

	// print how much time it took to sort the local maxima
	std::cout << "Sort time: " << double(sortEnd - sortStart) / CLOCKS_PER_SEC << std::endl;

	// extract the k best lines
	std::vector< std::pair<double, double> > kBest;

	// compute the stepsize in Theta- and r-dimension
	double stepSizeTheta = 1 / p.stepsPerRadian;
	double stepSizeR = 1 / p.stepsPerPixel;

	// take the k best lines from the sorted lines vector
	for (int i = 0; i < k; i++)
	{
		// compute Theta and r as real values (not the positions in the accumulator array!)
		double theta = p.minTheta + stepSizeTheta * maxima[i][0];
		double r = p.minR + stepSizeR * maxima[i][1];

		// add the line to the best lines vector
		kBest.push_back(std::make_pair(theta, r));
 	}

	return kBest;
}


// this methods draws one line in Theta-r-form with the given color in the passed image
void drawLine(CImg<unsigned char>& image, const double theta, const double r, double* color)
{
	// It's a rather horizontal line (cos(Theta) could be 0)
	if ((theta >= cimg::PI / 4 && theta <= cimg::PI * 3 / 4) || (theta >= cimg::PI * 5 / 4 && theta <= cimg::PI * 7 / 4))
	{
		// iterate horizontally over the image
		for (int x = 0; x < image.width(); x++)
		{
			// calculate y from r, Theta and x
			int y = round((r - x * cos(theta)) / sin(theta));

			// if y is within the range of the image, color the pixel (x,y)
			if (y >= 0 && y < image.height())
			{
				image(x,y,0,0) = color[0];
				image(x,y,0,1) = color[1];
				image(x,y,0,2) = color[2];
			}
		}
	}
	// it's a rather vertical line (sin(Theta) could be 0)
	else
	{
		// iterate vertically over the image
		for (int y = 0; y < image.height(); y++)
		{
			// calculate x from r, Theta and y
			int x = round((r - y * sin(theta)) / cos(theta));

			// if x is within the range of the image, color the pixel(x, y)
			if (x >= 0 && x < image.width())
			{
				image(x,y,0,0) = color[2];
				image(x,y,0,1) = color[1];
				image(x,y,0,2) = color[0];
			}
		}
	}
}


// this method receives some lines in Theta-r-form and draws them in the passed image
void drawLines(CImg<unsigned char>& image, std::vector< std::pair<double, double> >& lines, double* color)
{
	// iterate over lines
	for (int i = 0; i < int(lines.size()); i++)
	{
		// draw line
		drawLine(image, lines[i].first, lines[i].second, color);
	}
}


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

	// compute the binary image in the preprocess()-method and measure time
	clock_t preprocessStart = std::clock();
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

	CImg<bool> binaryImg = preprocess(filename.c_str(), thresholdDivisor);
	clock_t preprocessEnd = std::clock();

	// print how much time it took to compute the binary image
	std::cout << "Preprocess time: " << double(preprocessEnd - preprocessStart) / CLOCKS_PER_SEC << std::endl;

	// display the binary image
	CImgDisplay binaryImgDisp(binaryImg, "Binary Image");
	binaryImgDisp.move(50, 50);

	// save binary image as PNG-file
	(CImg<unsigned char> (binaryImg)).normalize(minColor, maxColor).save_png(std::string("binaryimg.png").insert(0, resultPath).c_str(), 1);

	HoughParameterSet p(binaryImg.width(), binaryImg.height());

	// compute the accumulator array and measure time
	clock_t houghStart = std::clock();
	CImg<long> accumulatorArray = computeAccumulatorArray(binaryImg, p);
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
	std::vector< std::pair<double, double> > best = getKBestLines(accumulatorArray, p, linesToExtract, excludeRadius);
	clock_t bestEnd = std::clock();

	// print how much time it took to compute the k best lines
	std::cout << "Best lines time: " << double(bestEnd - bestStart) / CLOCKS_PER_SEC << std::endl;

	// draw the lines and measure time
	CImg<unsigned char> bestLinesImg = binaryToColorImg(binaryImg);
	double redColor[] = {255, 0, 0};
	clock_t drawStart = std::clock();
	drawLines(bestLinesImg, best, redColor);
	clock_t drawEnd = std::clock();

	// print how long it took to draw the k best lines
	std::cout << "Draw time: " << double(drawEnd - drawStart) / CLOCKS_PER_SEC << std::endl;

	// save best line image as PNG-file
	(CImg<unsigned char> (bestLinesImg)).normalize(minColor, maxColor).save_png(std::string("bestlines.png").insert(0, resultPath).c_str(), 3);

	// display the best lines image
	CImgDisplay bestLinesDisp(bestLinesImg, "Best lines", 1);
	bestLinesDisp.move(0, 0);

	// Wait until display is closed
	while (!bestLinesDisp._is_closed)
		bestLinesDisp.wait();
}

