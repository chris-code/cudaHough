#pragma once

#include <vector>
#include <iostream>
#include <CImg.h>
#include <cuda_runtime.h>

using namespace cimg_library;

namespace cudaHough {
	bool * preprocess(CImg<double> image, double binarizationThreshold);
	CImg<long> transform(bool *binaryImage);
	std::vector<std::pair<double, double> > extractMostLikelyLines(CImg<long> accumulatorArray, long linesToExtract);

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
