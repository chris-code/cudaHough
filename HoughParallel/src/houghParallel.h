#pragma once

#include <vector>
#include <CImg.h>

using namespace std;

const long THREADS_PER_DIM = 8;

template<typename imgT>
cimg_library::CImg<imgT> gpuToCImg(imgT *image, long width, long height, bool freeMemory = true);

namespace cudaHough {
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

	template<typename imgT>
	bool * preprocess(cimg_library::CImg<imgT> &image, imgT binarizationThreshold, imgT sigma);

	template<typename accuT, typename paramT>
	accuT * transform(bool *binaryImage, long width, long height, HoughParameterSet<paramT> &hps);

	template<typename accuT, typename paramT>
	vector<pair<paramT, paramT> > extractStrongestLines(accuT *accumulatorArray, long linesToExtract,
		long excludeRadius, HoughParameterSet<paramT> &hps);

	template<typename imgT, typename accuT, typename paramT>
	vector<pair<paramT, paramT> > extractStrongestLines(cimg_library::CImg<imgT> &image,
		HoughParameterSet<paramT> &hps, imgT binarizationThreshold, imgT sigma, long linesToExtract,
		long excludeRadius);
}
