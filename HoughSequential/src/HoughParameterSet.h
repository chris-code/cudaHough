/*
 * HoughParameterSet.h
 *
 *  Created on: 27.12.2014
 *      Author: yannickubuntu
 */

#pragma once

#include <CImg.h>

class HoughParameterSet
{
	public:
		HoughParameterSet(int width, int height)
		{
			this->minTheta = 0;
			this->maxTheta = cimg_library::cimg::PI;
			this->stepsPerRadian = 57.295 * 2;
			this->stepsPerPixel = 2;
			this->minR = - sqrt(width * width + height * height);
			this->maxR = - minR;
		}

		virtual ~HoughParameterSet()
		{
		}

		double minTheta;
		double maxTheta;
		double stepsPerRadian;
		double stepsPerPixel;
		double minR;
		double maxR;
};
