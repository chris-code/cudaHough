#pragma once

#include <vector>
#include "CImg.h"

using namespace cimg_library;

namespace hough {
	template<typename paramT>
	class HoughParameterSet {
		public:
			HoughParameterSet(long width, long height) {
				this->minTheta = 0;
				this->maxTheta = cimg_library::cimg::PI;
				this->stepsPerRadian = 57.295 * 2;
				this->stepsPerPixel = 2.0;
				this->minR = -sqrt(width * width + height * height);
				this->maxR = -minR;
			}
			virtual ~HoughParameterSet() {
			}
			paramT getThetaStepSize() {
				return 1.0 / stepsPerRadian;
			}
			paramT getRstepSize() {
				return 1.0 / stepsPerPixel;
			}
			long getDimTheta() {
				return (maxTheta - minTheta) * stepsPerRadian + 1;
			}
			long getDimR() {
				return (maxR - minR) * stepsPerPixel + 1;
			}
			paramT minTheta;
			paramT maxTheta;
			paramT stepsPerRadian;
			paramT stepsPerPixel;
			paramT minR;
			paramT maxR;
	};

	template <typename imgT>
	CImg<bool> preprocess(CImg<imgT>& image, imgT thresholdDivisor);

	template<typename accuT, typename paramT>
	CImg<accuT> transform(CImg<bool>& binaryImg, hough::HoughParameterSet<paramT> & p);

	template<typename accuT, typename paramT>
	std::vector< std::pair<paramT, paramT> > extractStrongestLines(CImg<accuT>& accArray,
			hough::HoughParameterSet<paramT>& p, long k, long excludeRadius);
}
