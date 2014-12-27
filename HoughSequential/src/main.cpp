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
void printImg(const CImg<T>& imageE)
{
	for (int i = 0; i < imageE.width(); i++)
	{
		for (int j = 0; j < imageE.height(); j++)
		{
			std::cout << imageE(i, j, 0, 0) << "\t";
		}
		std::cout << std::endl;
	}
}


// this method transforms a rgb-color image to an grayvalue image
CImg<double> grayvalues(const CImg<double>& imageE)
{
	CImg<double> grayImg(imageE.width(), imageE.height(), 1, 1);

	double grayvalue = 0;
	for (int i = 0; i < imageE.width(); i++)
		for (int j = 0; j < imageE.height(); j++)
		{
			// The gray value is calculated by the following formula: 0.21 R + 0.72 G + 0.07 B
			grayvalue = 0.21 * imageE(i, j, 0, 0) + 0.72 * imageE(i, j, 0, 1) + 0.07 * imageE(i, j, 0, 2);
			grayImg(i,j,0,0) = grayvalue;
		}
	return grayImg;
}


// this methods normalizes an image so that the sum of its pixels equals 1
CImg<double> normalize(const CImg<double>& filterE)
{
	double sum;

	for (int i = 0; i < filterE.width(); i++)
	{
		for (int j = 0; j < filterE.height(); j++)
		{
			sum += filterE(i, j, 0, 0);
		}
	}

	CImg<double> normalizedFilter(filterE.width(), filterE.height());

	for (int i = 0; i < filterE.width(); i++)
	{
		for (int j = 0; j < filterE.height(); j++)
		{
			normalizedFilter(i, j, 0, 0) = filterE(i, j, 0, 0) / sum;
		}
	}

	return normalizedFilter;
}


// grayvalue Wrap-Around convolution
CImg<double> convolve(const CImg<double>& image, const CImg<double>& filter, const int offsetX, const int offsetY)
{
	CImg<double> convolvedImg(image.width(), image.height(), 1, 1);

	// iterate over image
	for (int imgX = 0; imgX < image.width(); imgX++)
	{
		for (int imgY = 0; imgY < image.height(); imgY++)
		{
			double value = 0;

			// iterate over filter
			for (int filX = 0; filX < filter.width(); filX++)
			{
				int posImgX = ((imgX - offsetX + filX) + image.width()) % image.width();

				for (int filY = 0; filY < filter.height(); filY++)
				{
					int posImgY = ((imgY - offsetY + filY) + image.height()) % image.height();

					value += image(posImgX, posImgY, 0, 0) * filter(filX, filY, 0, 0);
				}
			}

			convolvedImg(imgX, imgY, 0, 0) = value;
		}
	}

	return convolvedImg;
}


// calculates the gradient strength of an grayvalue image given the results of the convolution with SobelX and SobelY
CImg<double> calculateGradientStrength(const CImg<double>& sobelXE, const CImg<double>& sobelYE)
{
	CImg<double> strengthImg(sobelXE.width(), sobelXE.height(), 1, 1);

	for (int i = 0; i < sobelXE.width(); i++)
		for (int j = 0; j < sobelXE.height(); j++)
			strengthImg(i,j,0,0) = sqrt(sobelXE(i,j,0,0) * sobelXE(i,j,0,0) + sobelYE(i,j,0,0) * sobelYE(i,j,0,0));

	return strengthImg;
}


// returns a binary image given a grayvalue image
CImg<bool> makeBinaryImage(const CImg<double>& imageE, const double threshold)
{
	CImg<bool> binaryImg(imageE.width(), imageE.height(), 1, 1);

	for (int i = 0; i < imageE.width(); i++)
		for (int j = 0; j < imageE.height(); j++)
			if (imageE(i,j,0,0) > threshold)
				binaryImg(i,j,0,0) = true;
			else
				binaryImg(i,j,0,0) = false;

	return binaryImg;
}


CImg<bool> preprocess(const char* filename, double thresholdDivisor = 2)
{
	// load image from filename
	CImg<double> img(filename);

	// convert it to a grayvalue image
	CImg<double> grayImg = grayvalues(img);

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
//	double binomialArr[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
//	CImg<double> binomialUnnormalized(binomialArr, 3, 3);

	double bino1DArr[15] = {1, 2, 1};
	double bino2DArr[9];

	int c = 0;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			bino2DArr[c] = bino1DArr[i] * bino1DArr[j];
			c++;
		}
	}

	CImg<double> binomialUnnormalized(bino2DArr, 3, 3);
	CImg<double> binomial = normalize(binomialUnnormalized);

	// smooth image
	strengthImg = convolve(strengthImg, binomial, 4, 4);

	// calculate the binary image of the gradient strength image
	double threshold = (strengthImg.min() + strengthImg.max()) / thresholdDivisor;

	// make and return binary image
	return makeBinaryImage(strengthImg, threshold);
}

CImg<long> computeAccumulatorArray(const CImg<bool>& binaryImg, const HoughParameterSet& p)
{
	int dimTheta = (p.maxTheta - p.minTheta) * p.stepsPerRadian;
	int dimR = (p.maxR - p.minR) * p.stepsPerPixel;
	int borderExclude = 5;

	double thetaStepsize = 1 / p.stepsPerRadian;

	CImg<long> accumulatorArray(dimTheta, dimR, 1, 1, 0);

	for (int x = borderExclude; x < binaryImg.width() - borderExclude; x++)
	{
		for (int y = borderExclude; y < binaryImg.height() - borderExclude; y++)
		{
			if (binaryImg(x, y, 0, 0))
			{
				for (double theta = p.minTheta; theta <= p.maxTheta; theta += thetaStepsize)
				{
					double r = x * cos(theta) + y * sin(theta);
					int thetaIdx = int((theta - p.minTheta) * p.stepsPerRadian);
					int rIdx = int((r - p.minR) * p.stepsPerPixel);
					accumulatorArray(thetaIdx, rIdx, 0, 0) = accumulatorArray(thetaIdx, rIdx, 0, 0) +  1;
				}
			}
		}
	}

	return accumulatorArray;
}


CImg<unsigned char> binaryToColorImg(const CImg<bool>& binaryImg)
{
	CImg<unsigned char> colorImg(binaryImg.width(), binaryImg.height(), 1, 3, 0);

	for (int x = 0; x < binaryImg.width(); x++)
	{
		for (int y = 0; y < binaryImg.height(); y++)
		{
			if (binaryImg(x, y, 0, 0))
			{
				for (int k = 0; k < 3; k++)
					colorImg(x, y, 0, k) = 255;
			}
		}
	}

	return colorImg;
}

template <typename T>
std::vector< std::vector<int> > getLocalMaxima(const CImg<T>& image, int radius)
{
	std::vector< std::vector<int> > maxima;

	for (int x = 0; x < image.width(); x++)
	{
		for (int y = 0; y < image.height(); y++)
		{
			bool isMaximum = true;

			for (int i = -radius; i <= radius; i++)
			{
				int posX = ((x + i) + image.width()) % image.width();

				for (int j = -radius; j <= radius; j++)
				{
					int posY = ((y + j) + image.height()) % image.height();

					if (image(posX, posY, 0, 0) >= image(x, y, 0, 0) && (posX != x || posY != y))
						isMaximum = false;
				}
			}

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

	return maxima;
}

bool compareLines(std::vector<int> v1, std::vector<int> v2)
{
	return v1[2] > v2[2];
}

std::vector< std::pair<double, double> > getKBestLines(const CImg<long>& accArray, const HoughParameterSet& p, int k)
{
	// compute local maxima
	std::vector< std::vector<int> > maxima = getLocalMaxima(accArray, 10);

	// sort them
	std::sort(maxima.begin(), maxima.end(), compareLines);

	// extract the k best lines
	std::vector< std::pair<double, double> > kBest;

	double stepSizeTheta = 1 / p.stepsPerRadian;
	double stepSizeR = 1 / p.stepsPerPixel;

	for (int i = 0; i < k; i++)
	{
		double theta = p.minTheta + stepSizeTheta * maxima[i][0];
		double r = p.minR + stepSizeR * maxima[i][1];

		kBest.push_back(std::make_pair(theta, r));
 	}

	return kBest;
}


void drawLine(CImg<unsigned char>& image, const double theta, const double r, double* color)
{
	// it's not a vertical line
	if ((theta >= cimg::PI / 4 && theta <= cimg::PI * 3 / 4) || (theta >= cimg::PI * 5 / 4 && theta <= cimg::PI * 7 / 4))
	{
		for (int x = 0; x < image.width(); x++)
		{
			int y = round((r - x * cos(theta)) / sin(theta));

			if (y >= 0 && y < image.height())
			{
				image(x,y,0,0) = color[0];
				image(x,y,0,1) = color[1];
				image(x,y,0,2) = color[2];
			}
		}
	}
	// it may be a vertical line
	else
	{
		for (int y = 0; y < image.height(); y++)
		{
			int x = round((r - y * sin(theta)) / cos(theta));

			if (x >= 0 && x < image.width())
			{
				image(x,y,0,0) = color[2];
				image(x,y,0,1) = color[1];
				image(x,y,0,2) = color[0];
			}
		}
	}
}

void drawLines(CImg<unsigned char>& image, std::vector< std::pair<double, double> > lines)
{
	double color[3] = {255, 0, 0};

	for (int i = 0; i < lines.size(); i++)
	{
		drawLine(image, lines[i].first, lines[i].second, color);
//		color[0] = color[0] - 5;
	}
}


int main(int argc, char **argv)
{
	clock_t preprocessStart = std::clock();
	CImg<bool> binaryImg = preprocess("images/stoppschild3.jpg", 4);
	clock_t preprocessEnd = std::clock();

	CImgDisplay binaryImgDisp(binaryImg, "Binary Image");
	binaryImgDisp.move(50, 50);

	std::cout << "Preprocess time: " << double(preprocessEnd - preprocessStart) / CLOCKS_PER_SEC << std::endl;

	// Define parameters for the Hough Transformation
	double minTheta = 0;
	double maxTheta = 1 * cimg::PI;
	double stepsPerRadian = 57.295 * 2;
	double stepsPerPixel = 2;
	double maxR = sqrt(binaryImg.width() * binaryImg.width() + binaryImg.height() * binaryImg.height());
	double minR = -maxR;

	HoughParameterSet p(minTheta, maxTheta, stepsPerRadian, stepsPerPixel, minR, maxR);

	clock_t houghStart = std::clock();

	CImg<long> accumulatorArray = computeAccumulatorArray(binaryImg, p);

	clock_t houghEnd = std::clock();

	std::cout << "Hough time: " << double(houghEnd - houghStart) / CLOCKS_PER_SEC << std::endl;

	CImgDisplay accDisplay(accumulatorArray, "Accumulator Array", 1);
	accDisplay.move(400,50);

	clock_t drawStart = std::clock();

	CImg<unsigned char> bestLinesImg = binaryToColorImg(binaryImg);

	std::vector< std::pair<double, double> > best = getKBestLines(accumulatorArray, p, 16);
	drawLines(bestLinesImg, best);

	clock_t drawEnd = std::clock();

	std::cout << "Draw time: " << double(drawEnd - drawStart) / CLOCKS_PER_SEC << std::endl;

	CImgDisplay bestLinesDisp(bestLinesImg, "Best lines", 1);
	bestLinesDisp.move(0, 0);

	// Wait until display is closed
	while (!bestLinesDisp._is_closed)
		bestLinesDisp.wait();
}

