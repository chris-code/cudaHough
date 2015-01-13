#pragma once

#include <vector>
#include <CImg.h>

using namespace cimg_library;

// Transform an rgb-color image to a grayvalue image
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

// Create unsigned char color image from a binary image
template <typename T>
CImg<T> binaryToColorImg(const CImg<bool> &binaryImg) {
	// initialize the color image with the color black
	CImg<T> colorImg(binaryImg.width(), binaryImg.height(), 1, 3, 0);

	// iterate over the binary image
	for (long x = 0; x < binaryImg.width(); ++x) {
		for (long y = 0; y < binaryImg.height(); ++y) {
			// if there is a 1 in the binary image, set all three RGB-values to 255
			if (binaryImg(x, y, 0, 0)) {
				for (long k = 0; k < 3; ++k)
					colorImg(x, y, 0, k) = 255;
			}
		}
	}

	// return the color image
	return colorImg;
}

// this methods draws one line in Theta-r-form with the given color in the passed image
template <typename T, typename precisionT>
void drawLine(CImg<T> &image, const precisionT theta, const precisionT r, T *color) {
	// It's a rather horizontal line (cos(Theta) could be 0)
	if ((theta >= cimg::PI / 4 && theta <= cimg::PI * 3 / 4)
			|| (theta >= cimg::PI * 5 / 4 && theta <= cimg::PI * 7 / 4)) {
		// iterate horizontally over the image
		for (long x = 0; x < image.width(); ++x) {
			// calculate y from r, Theta and x
			long y = round((r - x * cos(theta)) / sin(theta));

			// if y is within the range of the image, color the pixel (x,y)
			if (y >= 0 && y < image.height()) {
				image(x, y, 0, 0) = color[0];
				image(x, y, 0, 1) = color[1];
				image(x, y, 0, 2) = color[2];
			}
		}
	}
	// it's a rather vertical line (sin(Theta) could be 0)
	else {
		// iterate vertically over the image
		for (long y = 0; y < image.height(); ++y) {
			// calculate x from r, Theta and y
			long x = round((r - y * sin(theta)) / cos(theta));

			// if x is within the range of the image, color the pixel(x, y)
			if (x >= 0 && x < image.width()) {
				image(x, y, 0, 0) = color[0];
				image(x, y, 0, 1) = color[1];
				image(x, y, 0, 2) = color[2];
			}
		}
	}
}

// this method receives some lines in Theta-r-form and draws them in the passed image
template <typename T, typename precisionT>
void drawLines(CImg<T> &image, std::vector<std::pair<precisionT, precisionT> > &lines, T *color) {
	// iterate over lines
	for (long i = 0; i < long(lines.size()); i++) {
		// draw line
		drawLine<T>(image, lines[i].first, lines[i].second, color);
	}
}
