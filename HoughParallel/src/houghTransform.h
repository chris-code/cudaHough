#pragma once

#include <vector>
#include <iostream>
#include <CImg.h>
#include <cuda_runtime.h>

using namespace cimg_library;

template<typename T>
CImg<T> gpuToCImg(T *image, long width, long height);

namespace cudaHough {
	template <typename T>
	bool * preprocess(CImg<T> &image, T binarizationThreshold);

	template <typename T>
	CImg<T> transform(bool *binaryImage);

	template <typename retT, typename paramT>
	std::vector<std::pair<retT, retT> > extractMostLikelyLines(CImg<paramT> &accumulatorArray, long linesToExtract);

	class HoughParameterSet {
		public:
			HoughParameterSet(int width, int height) {
				this->minTheta = 0;
				this->maxTheta = cimg_library::cimg::PI;
				this->stepsPerRadian = 57.295 * 2;
				this->stepsPerPixel = 2;
				this->minR = -sqrt(width * width + height * height);
				this->maxR = -minR;
			}

			virtual ~HoughParameterSet() {
			}

			double minTheta;
			double maxTheta;
			double stepsPerRadian;
			double stepsPerPixel;
			double minR;
			double maxR;
	};
}
