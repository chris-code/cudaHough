#include <CImg.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>

using namespace cimg_library;


// this methods prints the color values of an image
void printImg(const CImg<double>& imageE)
{
	for (int i = 0; i < imageE.width(); i++)
	{
		for (int j = 0; j < imageE.height(); j++)
		{
			std::cout << imageE(i, j, 0, 0) << " ";
		}
		std::cout << std::endl;
	}
}


// this method transforms a rgb-color image to an grayvalue image
CImg<double> grayvalues(const CImg<double>& imageE)
{
	CImg<double> grayImg(imageE._width, imageE._height, 1, 1);

	double grayvalue = 0;
	for (int i = 0; i < imageE.width(); i++)
		for (int j = 0; j < imageE.height(); j++)
			for (int k = 0; k < 3; k++)
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


// grayvalue Zero-padding convolution
CImg<double> convolve(const CImg<double>& imageE, const CImg<double>& filterE)
{
	// create new image that shall contain the convolved image
	CImg<double> convolvedImg(imageE.width(), imageE.height(), 1, 1, 0);

	// print error message, if the filter's height or width is odd
	if (filterE.height() % 2 == 0 || filterE.width() % 2 == 0)
	{
		std::cerr << "The filter's width and height have to be odd." << std::endl;
		return convolvedImg;
	}

	int halfFilW = filterE.width() / 2;
	int halfFilH = filterE.height() / 2;

	double tempSum;
	int imgW, imgH, filW, filH;

	// iterate over image
	for (int i = 0; i < imageE.width(); i++)
	{
		for (int j = 0; j < imageE.height(); j++)
		{
			tempSum = 0;

			// iterate over filter
			for (int fi = (-1) * halfFilW; fi <= halfFilW; fi++)
			{

				for (int fj = (-1) * halfFilH; fj <= halfFilH; fj++)
				{
					imgW = i + fi,
					imgH = j + fj,
					filW = fi + halfFilW,
					filH = fj + halfFilH;

					if (!(imgW < 0 || imgH < 0 || imgW > imageE.width() || imgH > imageE.height()))
						tempSum += imageE(imgW, imgH, 0, 0) * filterE(filW, filH, 0, 0);
				}
			}

			convolvedImg(i, j, 0, 0) = tempSum;
		}
	}
	return convolvedImg;
}


// calculates the gradient strength of an grayvalue image given the results of the convolution with SobelX and SobelY
CImg<double> calculateGradientStrength(const CImg<double>& sobelXE, const CImg<double>& sobelYE)
{
	CImg<double> strengthImg(sobelXE.width(), sobelYE.height(), 1, 1);

	for (int i = 0; i < sobelXE.width(); i++)
		for (int j = 0; j < sobelXE.height(); j++)
			strengthImg(i,j,0,0) = sqrt(sobelXE(i,j,0,0) * sobelXE(i,j,0,0) + sobelYE(i,j,0,0) * sobelYE(i,j,0,0));

	return strengthImg;
}


// returns maximal value of an grayvalue image
int getMaxValue(const CImg<double>& imageE)
{
	int max = INT_MIN;

	for (int i = 0; i < imageE.width(); i++)
		for (int j = 0; j < imageE.height(); j++)
			if (imageE(i,j,0,0) > max) max = imageE(i,j,0,0);

	return max;
}


// returns minimal value of an grayvalue image
int getMinValue(const CImg<double>& imageE)
{
	int min = INT_MAX;

	for (int i = 0; i < imageE.width(); i++)
		for (int j = 0; j < imageE.height(); j++)
			if (imageE(i,j,0,0) < min) min = imageE(i,j,0,0);

	return min;
}


// returns a binary image given a grayvalue image
CImg<double> makeBinaryImage(const CImg<double>& imageE)
{
	CImg<double> binaryImg(imageE.width(), imageE.height(), 1, 1);

	int max = getMaxValue(imageE);
	int min = getMinValue(imageE);

	int threshold = min + max;
	threshold /= 2;

	for (int i = 0; i < imageE.width(); i++)
		for (int j = 0; j < imageE.height(); j++)
			if (imageE(i,j,0,0) > threshold)
				binaryImg(i,j,0,0) = 1;
			else
				binaryImg(i,j,0,0) = 0;

	return binaryImg;
}


void houghTransform(const char* filename)
{
	std::cout << "Das Programm terminiert, wenn das Originalbild geschlossen wird." << std::endl;

	// load image from filename
	CImg<double> img(filename);

	// convert it to a grayvalue image
	CImg<double> grayImg = grayvalues(img);

	// show both images
	CImgDisplay mainDisp(img, "Originalbild", 0);
	CImgDisplay grayDisp(grayImg, "Grauwertbild", 0);

	// create Sobel X filter
	double sobelXarr[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	CImg<double> sobelX(sobelXarr, 3, 3);

	// create Sobel Y filter
	double sobelYarr[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	CImg<double> sobelY(sobelYarr, 3, 3);

	// convolve Image with both Sobel filters
	CImg<double> sobelXImg = convolve(grayImg, sobelX);
	CImg<double> sobelYImg = convolve(grayImg, sobelY);

	// display the results of the convolutions
	CImgDisplay sobelXDisp(sobelXImg, "Sobel X");
	CImgDisplay sobelYDisp(sobelYImg, "Sobel Y");

	// calculate the gradient strength
	CImg<double> strengthImg = calculateGradientStrength(sobelXImg, sobelYImg);
	CImgDisplay gradientStrengthDisp(strengthImg, "Gradientenstaerke");

	// calculate the binary image of the gradient strength image
	CImg<double> binaryImg = makeBinaryImage(strengthImg);
	CImgDisplay binaryImgDisp(binaryImg, "Binaerbild");

	// wait until the display with the original image is closed
	while (!mainDisp._is_closed)
		mainDisp.wait();
}


int main(int argc, char **argv)
{
	houghTransform("pidgey.jpg");
}

