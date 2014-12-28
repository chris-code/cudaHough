#include "houghTransform.h"

CImg<bool> cudaHough::preprocess(CImg<double> image) {
//	TODO
}

CImg<long> cudaHough::transform(CImg<bool> binaryImage) {
//	TODO
}

std::vector<std::pair<double, double> > cudaHough::extractMostLikelyLines(
		CImg<long> accumulatorArray, long linesToExtract) {

}
