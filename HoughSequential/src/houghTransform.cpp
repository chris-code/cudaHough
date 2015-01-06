#include <algorithm>
#include "houghTransform.h"


// this methods convolves an gray value image with a filter using the Wrap-Around approach
template <typename imgT>
CImg<imgT> convolve(const CImg<imgT>& image, const CImg<imgT>& filter, const int offsetX, const int offsetY)
{
	// initialize the convolved image
	CImg<imgT> convolvedImg(image.width(), image.height(), 1, 1);

	// iterate over image
	for (int imgX = 0; imgX < image.width(); imgX++)
	{
		for (int imgY = 0; imgY < image.height(); imgY++)
		{
			// set the temporary sum to 0
			imgT tempSum = 0;

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
template <typename imgT>
CImg<imgT> calculateGradientStrength(const CImg<imgT>& sobelX, const CImg<imgT>& sobelY)
{
	// initialize the gradient strength image
	CImg<imgT> strengthImg(sobelX.width(), sobelX.height(), 1, 1);

	// iterate over the strength image
	for (int x = 0; x < sobelX.width(); x++)
		for (int y = 0; y < sobelX.height(); y++)
			// calculate the strength of the gradient at position (x,y) from the values in sobelX(x,y) and sobelY(x,y)
			strengthImg(x,y,0,0) = sqrt(sobelX(x,y,0,0) * sobelX(x,y,0,0) + sobelY(x,y,0,0) * sobelY(x,y,0,0));

	// return the strength image
	return strengthImg;
}


// this methods normalizes an image so that the sum of its pixels equals 1
template <typename imgT>
CImg<imgT> normalize(const CImg<imgT>& filter)
{
	imgT sum = 0.;

	// get the sum of all filter values
	for (int i = 0; i < filter.width(); i++)
	{
		for (int j = 0; j < filter.height(); j++)
		{
			sum += filter(i, j, 0, 0);
		}
	}

	// initialize the normalized filter
	CImg<imgT> normalizedFilter(filter.width(), filter.height());

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


// returns a binary image given a grayvalue image
template <typename imgT>
CImg<bool> makeBinaryImage(const CImg<imgT>& image, const imgT threshold)
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
template <typename imgT>
CImg<bool> hough::preprocess(CImg<imgT>& image, imgT relativeThreshold)
{
	// create Sobel X filter
	imgT sobelXarr[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	CImg<imgT> sobelX(sobelXarr, 3, 3);

	// create Sobel Y filter
	imgT sobelYarr[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	CImg<imgT> sobelY(sobelYarr, 3, 3);

	// convolve Image with both Sobel filters
	CImg<imgT> sobelXImg = convolve<imgT>(image, sobelX, 1, 1);
	CImg<imgT> sobelYImg = convolve<imgT>(image, sobelY, 1, 1);

	// calculate the gradient strength
	CImg<imgT> strengthImg = calculateGradientStrength<imgT>(sobelXImg, sobelYImg);

	// create binomial filter
	imgT binomialArr[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	CImg<imgT> binomialUnnormalized(binomialArr, 3, 3);
	CImg<imgT> binomial = normalize<imgT>(binomialUnnormalized);

	// smooth image
	strengthImg = convolve<imgT>(strengthImg, binomial, 4, 4);

	// calculate the binary image of the gradient strength image
	imgT strengthImgMin = strengthImg.min();
	imgT strengthImgMax = strengthImg.max();
	imgT absoluteThreshold = (strengthImgMax - strengthImgMin) * relativeThreshold + strengthImgMin;

	// make and return binary image
	return makeBinaryImage<imgT>(strengthImg, absoluteThreshold);
}


// this methods computes the accumulator array
template<typename accuT, typename paramT>
CImg<accuT> hough::transform(CImg<bool>& binaryImg, hough::HoughParameterSet<paramT> & p)
{
	// calculate the dimensions of the accumulator array by the given HoughParameterSet
	int dimTheta = p.getDimTheta();
	int dimR = p.getDimR();

	// initialize the border exclude value, which defines how much of the border is not considered
	int borderExclude = 5;

	// calculate the thetaStepsize by inverting the steps per radian

	// initialize the accumulator array as black image (initially every line has 0 votes)
	CImg<accuT> accumulatorArray(dimTheta, dimR, 1, 1, 0);

	// iterate over the image and ignore some points at the border
	for (int x = borderExclude; x < binaryImg.width() - borderExclude; x++)
	{
		for (int y = borderExclude; y < binaryImg.height() - borderExclude; y++)
		{
			// if there is written a 1 in the binary image, increment the vote matrix at the corresponding positions
			if (binaryImg(x, y, 0, 0))
			{
				// iterate over all possible values for Theta
				for (paramT theta = p.minTheta; theta <= p.maxTheta; theta += p.getThetaStepSize())
				{
					// calculate the r value
					paramT r = x * cos(theta) + y * sin(theta);

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


// this methods finds local optima in an image
template <typename imgT>
std::vector< std::vector<int> > getLocalMaxima(const CImg<imgT>& image, int excludeRadius)
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
					{
						isMaximum = false;
						i = excludeRadius + 1; // TODO find nicer method to end loops
						j = excludeRadius + 1;
					}
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
template<typename accuT, typename paramT>
std::vector< std::pair<paramT, paramT> > hough::extractStrongestLines(CImg<accuT>& accArray,
		hough::HoughParameterSet<paramT>& p, int k, int excludeRadius)
{
	// compute local maxima
	std::vector< std::vector<int> > maxima = getLocalMaxima(accArray, excludeRadius);

	// sort them
	std::sort(maxima.begin(), maxima.end(), compareLines);

	// extract the k best lines
	std::vector< std::pair<paramT, paramT> > kBest;

	// take the k best lines from the sorted lines vector
	for (int i = 0; i < k; i++)
	{
		// compute Theta and r as real values (not the positions in the accumulator array!)
		paramT theta = p.minTheta + p.getThetaStepSize() * maxima[i][0];
		paramT r = p.minR + p.getRstepSize() * maxima[i][1];

		// add the line to the best lines vector
		kBest.push_back(std::make_pair(theta, r));
 	}

	return kBest;
}

// Instantiate template methods so they are available to the compiler
template CImg<bool> hough::preprocess(CImg<float>& image, float thresholdDivisor);
template CImg<bool> hough::preprocess(CImg<double>& image, double thresholdDivisor);
template CImg<long> hough::transform(CImg<bool>& binaryImg, hough::HoughParameterSet<float> & p);
template CImg<long> hough::transform(CImg<bool>& binaryImg, hough::HoughParameterSet<double> & p);
template std::vector< std::pair<float, float> > hough::extractStrongestLines(CImg<long>& accArray,
		hough::HoughParameterSet<float>& p, int k, int excludeRadius);
template std::vector< std::pair<double, double> > hough::extractStrongestLines(CImg<long>& accArray,
		hough::HoughParameterSet<double>& p, int k, int excludeRadius);
