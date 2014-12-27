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
CImg<double> makeBinaryImage(const CImg<double>& imageE, const double threshold)
{
	CImg<double> binaryImg(imageE.width(), imageE.height(), 1, 1);

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
	mainDisp.move(50, 50);
	CImgDisplay grayDisp(grayImg, "Grauwertbild", 0);
	grayDisp.move(400, 50);

	// create Sobel X filter
	double sobelXarr[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	CImg<double> sobelX(sobelXarr, 3, 3);

	// create Sobel Y filter
	double sobelYarr[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	CImg<double> sobelY(sobelYarr, 3, 3);

	// convolve Image with both Sobel filters
	CImg<double> sobelXImg = convolve(grayImg, sobelX, 1, 1);
	CImg<double> sobelYImg = convolve(grayImg, sobelY, 1, 1);

	// display the results of the convolutions
	CImgDisplay sobelXDisp(sobelXImg, "Sobel X");
	sobelXDisp.move(750, 50);
	CImgDisplay sobelYDisp(sobelYImg, "Sobel Y");
	sobelYDisp.move(50,430);

	// calculate the gradient strength
	CImg<double> strengthImg = calculateGradientStrength(sobelXImg, sobelYImg);
	CImgDisplay gradientStrengthDisp(strengthImg, "Gradientenstaerke");
	gradientStrengthDisp.move(400, 430);

	// calculate the binary image of the gradient strength image
	double min, max;
	max = strengthImg.min_max(min);
	double threshold = (min + max) / 2;
	CImg<double> binaryImg = makeBinaryImage(strengthImg, threshold);
	CImgDisplay binaryImgDisp(binaryImg, "Binaerbild");
	binaryImgDisp.move(750, 430);

	// wait until the display with the original image is closed
	while (!mainDisp._is_closed)
		mainDisp.wait();
}


int main(int argc, char **argv)
{
	houghTransform("pidgey.jpg");
	houghTransform("charizard.jpg");
}

