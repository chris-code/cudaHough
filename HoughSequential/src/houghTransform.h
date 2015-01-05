#pragma once

namespace hough {
	template<typename paramT>
	class HoughParameterSet {
		public:
			HoughParameterSet(int width, int height) {
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
}
