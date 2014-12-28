#include "houghTransform.h"

// this method transforms a rgb-color image to an grayvalue image
CImg<double> RGBToGrayValueImage(const CImg<double> &image) {
	// initialize the gray value image
	CImg<double> grayImg(image.width(), image.height(), 1, 1);

	// iterate over the image
	for (int i = 0; i < image.width(); i++)
		for (int j = 0; j < image.height(); j++) {
			// The gray value is calculated by the following formula: 0.21 R + 0.72 G + 0.07 B
			grayImg(i, j, 0, 0) = 0.21 * image(i, j, 0, 0) + 0.72 * image(i, j, 0, 1) + 0.07 * image(i, j, 0, 2);
		}

	return grayImg;
}

#define assertCheck(arg) { errorCheck(arg, __FILE__, __LINE__); }
void errorCheck(cudaError_t returnCode, char *file, long line) {
	if (returnCode != cudaSuccess) {
		std::cerr << cudaGetErrorString(returnCode) << " occured at " << line << " in file " << file << std::endl;
		exit(EXIT_FAILURE);
	}
}

double * cImgToGPU(CImg<double> image) {
	double *gpuImage;
	assertCheck( cudaMalloc(&gpuImage, image.height() * image.width() * sizeof(double)) );
	assertCheck(
			cudaMemcpy(gpuImage, image.data(), image.height() * image.width() * sizeof(double), cudaMemcpyHostToDevice));
	return gpuImage;
}

CImg<double> gpuToCImg(double *image, long width, long height) {
	double *cpuData = new double[width * height];
	assertCheck( cudaMemcpy(&cpuData, image, width * height * sizeof(double), cudaMemcpyDeviceToHost) );
	CImg<double> cpuImg(cpuData, width, height);
	delete[] cpuData;
	return cpuImg;
}

__global__ void convolve(double *result, double *image, long imgWidth, long imgHeight, double *filter, long filWidth,
		long filHeight, long filAnchorX, long filAnchorY) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < imgWidth && y < imgHeight) {
		double value = 0;

		for (long filX = 0; filX < filWidth; ++filX) {
			long posImgX = ((x - filAnchorX + filX) + imgWidth) % imgWidth;
			for (long filY = 0; filY < filHeight; ++filY) {
				long posImgY = ((y - filAnchorY + filY) + imgHeight) % imgHeight;

				value += image[posImgY * imgWidth + posImgX] * filter[filY * filWidth + filX];
			}
		}

		image[y * imgWidth + x] = value;
	}
}

__global__ void computeGradientStrengthGPU(double *gradientStrength, double *gradientX, double *gradientY, long width,
		long height) {
	long x = blockIdx.x * blockDim.x + threadIdx.x;
	long y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		long index = y * width + x;
		gradientStrength[index] = sqrt(pow(gradientX[index], 2) + pow(gradientY[index], 2));
	}
}

double * computeGradientStrength(double *grayValueImage, long width, long height) {
	double cpuSobelX[] = { 1, 0, -1, 2, 0, -2, 1, 0, -1 };
	double cpuSobelY[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	double *sobelX;
	double *sobelY;
	assertCheck(cudaMalloc(&sobelX, 9 * sizeof(double)));
	assertCheck(cudaMalloc(&sobelY, 9 * sizeof(double)));
	assertCheck(cudaMemcpy(sobelX, cpuSobelX, 9 * sizeof(double), cudaMemcpyHostToDevice));
	assertCheck(cudaMemcpy(sobelY, cpuSobelY, 9 * sizeof(double), cudaMemcpyHostToDevice));

	double *gradientX;
	double *gradientY;
	assertCheck(cudaMalloc(&gradientX, width * height * sizeof(double)));
	assertCheck(cudaMalloc(&gradientY, width * height * sizeof(double)));

	dim3 threads(16, 16);
	dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
	convolve<<<blocks, threads>>>(gradientX, grayValueImage, width, height, sobelX, 3, 3, 1, 1);
	convolve<<<blocks, threads>>>(gradientY, grayValueImage, width, height, sobelY, 3, 3, 1, 1);

	double *gradientStrength;
	assertCheck(cudaMalloc(&gradientStrength, width * height * sizeof(double)));
	computeGradientStrengthGPU<<<blocks, threads>>>(gradientStrength, gradientX, gradientY, width, height);

	cudaFree(sobelX);
	cudaFree(sobelY);
	cudaFree(gradientX);
	cudaFree(gradientY);

	return gradientStrength;
}

CImg<bool> cudaHough::preprocess(CImg<double> image) {
	CImg<double> cpuGrayValueImage = RGBToGrayValueImage(image);
	double *grayValueImage = cImgToGPU(cpuGrayValueImage);

	double *gradientStrengthImage = computeGradientStrength(grayValueImage, image.width(), image.height());

	CImg<double> CPUgradientStrenthImage = gpuToCImg(gradientStrengthImage, image.width(), image.height());
	CImgDisplay d(CPUgradientStrenthImage, "foo", 1);
	while(!d.is_closed()) {
		d.wait();
	}

	cudaFree(grayValueImage);
}

CImg<long> cudaHough::transform(CImg<bool> binaryImage) {
//	TODO
}

std::vector<std::pair<double, double> > cudaHough::extractMostLikelyLines(CImg<long> accumulatorArray,
		long linesToExtract) {

}
