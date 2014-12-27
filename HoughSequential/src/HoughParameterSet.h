/*
 * HoughParameterSet.h
 *
 *  Created on: 27.12.2014
 *      Author: yannickubuntu
 */

#pragma once

class HoughParameterSet
{
	public:
		HoughParameterSet(double minTheta, double maxTheta, double stepsPerRadian, double stepsPerPixel, double minR, double maxR)
		{
			this->minTheta = minTheta;
			this->maxTheta = maxTheta;
			this->stepsPerRadian = stepsPerRadian;
			this->stepsPerPixel = stepsPerPixel;
			this->minR = minR;
			this->maxR = maxR;
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
