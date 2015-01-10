#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "houghTransform.h"

// this method transforms a rgb-color image to an grayvalue image
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

#define assertCheck(arg) { errorCheck((arg), __FILE__, __LINE__); }
void errorCheck(const cudaError_t returnCode, const char *file, const long line) {
	if (returnCode != cudaSuccess) {
		std::cerr << cudaGetErrorString(returnCode) << " occurred at " << line << " in file " << file << std::endl;
		exit(EXIT_FAILURE);
	}
}

template<typename T>
T * cImgToGPU(CImg<T> &image) {
	T *gpuImage;
	assertCheck(cudaMalloc(&gpuImage, image.height() * image.width() * sizeof(T)));
	assertCheck(cudaMemcpy(gpuImage, image.data(), image.height() * image.width() * sizeof(T), cudaMemcpyHostToDevice));
	return gpuImage;
}

template<typename imgT>
CImg<imgT> gpuToCImg(imgT *image, long width, long height, bool freeMemory) {
	imgT *cpuData = (imgT*) malloc(width * height * sizeof(imgT));
	assertCheck(cudaMemcpy(cpuData, image, width * height * sizeof(imgT), cudaMemcpyDeviceToHost));
	if (freeMemory)
		assertCheck(cudaFree(image));
	CImg<imgT> cpuImg(cpuData, width, height);
	free(cpuData);
	return cpuImg;
}

template<typename T>
__global__ void convolve(T *result, T *image, long imgWidth, long imgHeight, T *filter, long filWidth, long filHeight,
	long filAnchorX, long filAnchorY) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < imgWidth && y < imgHeight) {
		T value = 0;

		for (long filX = 0; filX < filWidth; ++filX) {
			long posImgX = ((x - filAnchorX + filX) + imgWidth) % imgWidth;
			for (long filY = 0; filY < filHeight; ++filY) {
				long posImgY = ((y - filAnchorY + filY) + imgHeight) % imgHeight;

				value += image[posImgY * imgWidth + posImgX] * filter[filY * filWidth + filX];
			}
		}

		result[y * imgWidth + x] = value;
	}
}

//	Note that this kernel takes sigma^2 as an argument.
//	TODO this kernel could actually just take one dimension instead of width and height. See if this would run faster.
template<typename imgT>
__global__ void generateGauss(imgT *result, long width, long height, imgT sigma2, imgT normalizationTerm) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;
	long index = y * width + x;
	if (x < width && y < height) {
		imgT coordX = x - width / 2.0;
		imgT coordY = y - height / 2.0;

		imgT value = pow(coordX, imgT(2.0)) + pow(coordY, imgT(2.0));
		value /= 2 * sigma2;
		value = exp(-value);
		result[index] = value * normalizationTerm;
	}
}

template<typename imgT>
imgT * gaussBlurr(imgT *image, long width, long height, imgT sigma) {
	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

//	Generate gaussian
	long filterWidth = 2 * sigma + 1; // 2-sigma rule, catch 95% of all values
	long filterHeight = 2 * sigma + 1; // (but make it odd so that a center exists)
//	TODO we get about 95% of all values. The gauss filter would normally sum to 1 since the gauss curve integrates to
//	1. But we lose 5% of it, so it doesn't exactly sum to one. Modify the normalization term to compensate for this.
	imgT normalizationTerm = 1.0 / (2.0 * M_PI * pow(sigma, 2.0));
	imgT *gauss;
	assertCheck(cudaMalloc(&gauss, filterWidth * filterHeight * sizeof(imgT)));
	generateGauss<imgT> <<<blocks, threads>>>(gauss, filterWidth, filterHeight, imgT(pow(sigma, 2.0)),
		imgT(normalizationTerm));
	assertCheck(cudaGetLastError());

//	Blurr the image with it
	imgT *result;
	assertCheck(cudaMalloc(&result, width * height * sizeof(imgT)));
	convolve<imgT> <<<blocks, threads>>>(result, image, width, height, gauss, filterWidth, filterHeight,
		filterWidth / 2, filterHeight / 2);
	assertCheck(cudaGetLastError());
	assertCheck(cudaFree(gauss));

	return result;
}

template<typename T>
__global__ void computeGradientStrengthGPU(T *gradientStrength, T *gradientX, T *gradientY, long width, long height) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		long index = y * width + x;
		gradientStrength[index] = sqrt(pow(gradientX[index], 2) + pow(gradientY[index], 2));
	}
}

template<typename T>
T * computeGradientStrength(T *grayValueImage, long width, long height) {
	T cpuSobelX[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	T cpuSobelY[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
	T *sobelX;
	T *sobelY;
	assertCheck(cudaMalloc(&sobelX, 9 * sizeof(T)));
	assertCheck(cudaMalloc(&sobelY, 9 * sizeof(T)));
	assertCheck(cudaMemcpy(sobelX, cpuSobelX, 9 * sizeof(T), cudaMemcpyHostToDevice));
	assertCheck(cudaMemcpy(sobelY, cpuSobelY, 9 * sizeof(T), cudaMemcpyHostToDevice));

	T *gradientX;
	T *gradientY;
	assertCheck(cudaMalloc(&gradientX, width * height * sizeof(T)));
	assertCheck(cudaMalloc(&gradientY, width * height * sizeof(T)));

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	convolve<T> <<<blocks, threads>>>(gradientX, grayValueImage, width, height, sobelX, 3, 3, 1, 1);
	assertCheck(cudaGetLastError());
	convolve<T> <<<blocks, threads>>>(gradientY, grayValueImage, width, height, sobelY, 3, 3, 1, 1);
	assertCheck(cudaGetLastError());

	T *gradientStrength;
	assertCheck(cudaMalloc(&gradientStrength, width * height * sizeof(T)));
	computeGradientStrengthGPU<T> <<<blocks, threads>>>(gradientStrength, gradientX, gradientY, width, height);
	assertCheck(cudaGetLastError());

	assertCheck(cudaFree(sobelX));
	assertCheck(cudaFree(sobelY));
	assertCheck(cudaFree(gradientX));
	assertCheck(cudaFree(gradientY));

	return gradientStrength;
}

template<typename T>
__global__ void binarizeGPU(bool *result, T *image, long width, long height, T threshold) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		long index = y * width + x;
		if (image[index] > threshold)
			result[index] = 1;
		else
			result[index] = 0;
	}
}

template<typename T>
bool * binarize(T *image, long width, long height, T relativeThreshold) {
	thrust::device_ptr<T> imageThrustPointer(image);
	T min = (thrust::min_element(imageThrustPointer, imageThrustPointer + (width * height)))[0];
	T max = (thrust::max_element(imageThrustPointer, imageThrustPointer + (width * height)))[0];
	T absoluteThreshold = (max - min) * relativeThreshold + min;

	bool *binaryImage;
	assertCheck(cudaMalloc(&binaryImage, width * height * sizeof(bool)));

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	binarizeGPU<T> <<<blocks, threads>>>(binaryImage, image, width, height, absoluteThreshold);
	assertCheck(cudaGetLastError());

	return binaryImage;
}

template<typename imgT>
bool * cudaHough::preprocess(CImg<imgT> &image, imgT relativeThreshold, imgT sigma) {
	imgT *grayValueImage = cImgToGPU<imgT>(image);
	imgT *blurredImage = gaussBlurr<imgT>(grayValueImage, image.width(), image.height(), sigma);
	imgT *gradientStrengthImage = computeGradientStrength<imgT>(blurredImage, image.width(), image.height());
	bool *binaryImage = binarize<imgT>(gradientStrengthImage, image.width(), image.height(), relativeThreshold);

	assertCheck(cudaFree(grayValueImage));
	assertCheck(cudaFree(blurredImage));
	assertCheck(cudaFree(gradientStrengthImage));

	return binaryImage;
}

template<typename accuT, typename Tparam>
__global__ void computeAccumulatorArrayGPU(bool *binaryImage, long width, long height, long borderExclude,
	accuT *accumulatorArray, Tparam minTheta, Tparam maxTheta, Tparam thetaStepSize, Tparam stepsPerRadian,
	Tparam minR, Tparam stepsPerPixel, long dimTheta) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;
//	TODO calculate x and y by directly taking into account border exclude, instead of checking it afterwards
	if (x >= borderExclude && y >= borderExclude && x < width - borderExclude && y < height - borderExclude) {
		if (binaryImage[y * width + x] == 1) {
			for (Tparam theta = minTheta; theta < maxTheta; theta += thetaStepSize) {
				Tparam r = x * cos(theta) + y * sin(theta);

				long thetaIdx = long((theta - minTheta) * stepsPerRadian);
				long rIdx = long((r - minR) * stepsPerPixel);
				accumulatorArray[rIdx * dimTheta + thetaIdx] += 1;
			}
		}
	}
}

template<typename accuT, typename paramT>
accuT * cudaHough::transform(bool *binaryImage, long width, long height, HoughParameterSet<paramT> &hps) {
	long dimTheta = hps.getDimTheta();
	long dimR = hps.getDimR();
	long borderExclude = 5;

	accuT *accumulatorArray;
	assertCheck(cudaMalloc(&accumulatorArray, dimTheta * dimR * sizeof(accuT)));
	cudaMemset(accumulatorArray, 0, dimTheta * dimR * sizeof(accuT));

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	computeAccumulatorArrayGPU<accuT, paramT> <<<blocks, threads>>>(binaryImage, width, height, borderExclude,
		accumulatorArray, hps.minTheta, hps.maxTheta, hps.getThetaStepSize(), hps.stepsPerRadian, hps.minR,
		hps.stepsPerPixel, dimTheta);
	assertCheck(cudaGetLastError());

	return accumulatorArray;
}

template<typename T>
__global__ void isolateLocalMaximaGPU(T *accumulatorArray, T *localMaxima, long width, long height,
	long excludeRadius) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		long index = y * width + x;

		for (long offsetX = -excludeRadius; offsetX <= excludeRadius; ++offsetX) {
			long posX = ((x + offsetX) + width) % width;

			for (long offsetY = -excludeRadius; offsetY <= excludeRadius; ++offsetY) {
				long posY = ((y + offsetY) + height) % height;
				long offsetIndex = ((posY * width) + posX);

				if (accumulatorArray[offsetIndex] >= accumulatorArray[index] && offsetIndex != index) {
					localMaxima[index] = -1;
					return;
				}
			}
		}

		localMaxima[index] = accumulatorArray[index];
	}
}

template<typename T>
T * isolateLocalMaxima(T *accumulatorArray, long width, long height, long excludeRadius) {
	T *localMaxima;
	assertCheck(cudaMalloc(&localMaxima, width * height * sizeof(T)));

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	isolateLocalMaximaGPU<T> <<<blocks, threads>>>(accumulatorArray, localMaxima, width, height, excludeRadius);
	assertCheck(cudaGetLastError());

	return localMaxima;
}

template<typename T>
thrust::device_vector<long> getSortedIndices(T *maxima, long width, long height) {
	thrust::device_vector<long> indices(width * height);
	thrust::sequence(indices.begin(), indices.end());

	thrust::device_ptr<T> maximaThrust(maxima);
	thrust::sort_by_key(maximaThrust, maximaThrust + width * height, indices.begin(), thrust::greater<long>());

	return indices; // TODO don't return thrust::device_vector
}

template<typename accuT, typename paramT>
std::vector<std::pair<paramT, paramT> > cudaHough::extractStrongestLines(accuT *accumulatorArray, long linesToExtract,
	long excludeRadius, HoughParameterSet<paramT> &hps) {
	accuT *localMaxima = isolateLocalMaxima(accumulatorArray, hps.getDimTheta(), hps.getDimR(), excludeRadius);
	thrust::device_vector<long> sortedIndices = getSortedIndices<accuT>(localMaxima, hps.getDimTheta(), hps.getDimR());
	thrust::host_vector<long> cpuSortedIndices(linesToExtract);
	thrust::copy(sortedIndices.begin(), sortedIndices.begin() + linesToExtract, cpuSortedIndices.begin());

	std::vector<std::pair<paramT, paramT> > bestLines;
	for (long i = 0; i < linesToExtract; i++) {
		long x = cpuSortedIndices[i] % hps.getDimTheta();
		long y = cpuSortedIndices[i] / hps.getDimTheta();

		double theta = hps.minTheta + hps.getThetaStepSize() * x;
		double r = hps.minR + hps.getRstepSize() * y;

		bestLines.push_back(std::make_pair<double, double>(theta, r));
	}
	assertCheck(cudaFree(localMaxima));

	return bestLines;
}

template<typename imgT, typename accuT, typename paramT>
std::vector<std::pair<paramT, paramT> > cudaHough::extractStrongestLines(CImg<imgT> &image,
	HoughParameterSet<paramT> &hps, imgT binarizationThreshold, imgT sigma, long linesToExtract, long excludeRadius) {
	bool *binaryImage = preprocess<imgT>(image, binarizationThreshold, sigma);
	accuT *accumulatorArray = transform<accuT, paramT>(binaryImage, image.width(), image.height(), hps);
	std::vector<std::pair<paramT, paramT> > strongestLines = extractStrongestLines<accuT, paramT>(accumulatorArray,
		linesToExtract, excludeRadius, hps);

	assertCheck(cudaFree(binaryImage));
	assertCheck(cudaFree(accumulatorArray));

	return strongestLines;
}

// Instantiate template methods so they are available to the compiler
template CImg<bool> gpuToCImg(bool *image, long width, long height, bool freeMemory);
template CImg<long> gpuToCImg(long *image, long width, long height, bool freeMemory);
template CImg<float> gpuToCImg(float *image, long width, long height, bool freeMemory);
template CImg<double> gpuToCImg(double *image, long width, long height, bool freeMemory);
template bool * cudaHough::preprocess<float>(CImg<float> &image, float threshold, float sigma);
template bool * cudaHough::preprocess<double>(CImg<double> &image, double threshold, double sigma);
template long * cudaHough::transform(bool *binaryImage, long width, long height, HoughParameterSet<float> &hps);
template long * cudaHough::transform(bool *binaryImage, long width, long height, HoughParameterSet<double> &hps);
template std::vector<std::pair<float, float> > cudaHough::extractStrongestLines(long *accumulatorArray,
	long linesToExtract, long excludeRadius, HoughParameterSet<float> &hps);
template std::vector<std::pair<double, double> > cudaHough::extractStrongestLines(long *accumulatorArray,
	long linesToExtract, long excludeRadius, HoughParameterSet<double> &hps);
template std::vector<std::pair<float, float> > cudaHough::extractStrongestLines<float, long, float>(
	CImg<float> &image, HoughParameterSet<float> &hps, float threshold, float sigma, long linesToExtract,
	long excludeRadius);
template std::vector<std::pair<double, double> > cudaHough::extractStrongestLines<double, long, double>(
	CImg<double> &image, HoughParameterSet<double> &hps, double threshold, double sigma, long linesToExtract,
	long excludeRadius);
