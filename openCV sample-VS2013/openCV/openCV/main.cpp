#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"
#include "math.h"
using namespace std;
using namespace cv;

float grayLevel[800][500];
float grayLevelx[800][500];
float grayLevely[800][500];
float grayLevelA[800][500];

float averagingMask[5][5];

void Laplacian(Mat &src, Mat &dst)
{

	float mask[3][3] = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
	float min = 1000;
	float max = -1000;

	//apply filter
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (i == 0 || j == 0 || i == src.rows - 1 || j == src.cols - 1)
			{
				grayLevel[i][j] = 0.0;
				
			}
			else 
			{
				grayLevel[i][j] = src.at<uchar>(i - 1, j)*mask[0][1] + src.at<uchar>() + src.at<uchar>(i, j - 1)*mask[1][0] + src.at<uchar>(i, j + 1)*mask[1][2] + src.at<uchar>(i + 1, j)*mask[2][1] + src.at<uchar>(i, j)*mask[1][1] + src.at<uchar>(i - 1, j - 1)*mask[0][0] + src.at<uchar>(i + 1, j - 1)*mask[2][0] + src.at<uchar>(i - 1, j + 1)*mask[0][2] + src.at<uchar>(i + 1, j + 1)*mask[2][2];
			}
			if (grayLevel[i][j] < min)
			{
				min = grayLevel[i][j];
			}
		}
	}

	//fm
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = grayLevel[i][j] - min;

			if (grayLevel[i][j] > max) 
			{
				max = grayLevel[i][j];
			}
		}
	}

	//assign fs to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 * (grayLevel[i][j] / max);
		}
	}
}

void SrcPlusLaplacian(const Mat &src, Mat &dst, Mat &lap) 
{
	float min = 100000;
	float max = -100000;

	//SrcPlusLaplacian
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = src.at<uchar>(i, j) + lap.at<uchar>(i, j);

			if (grayLevel[i][j] < min)
			{
				min = grayLevel[i][j];
			}
		}
	}

	//fm
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = grayLevel[i][j] - min;

			if (grayLevel[i][j] > max)
			{
				max = grayLevel[i][j];
			}
		}
	}

	//assign fs to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 * (grayLevel[i][j] / max);
		}
	}
	
}
		
void Averaging(const Mat &src, Mat &dst)
{
	//initiate averagingMask
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			averagingMask[i][j] = 0.04;
		}
	}
	
	//initiate grayLevelA
	for (int i = 0; i < 800; i++)
	{
		for (int j = 0; j < 500; j++)
		{
			grayLevelA[i][j] = 0;
		}
	}

	//applying filter
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (i == 0 || j == 0 || i == src.rows - 1 || j == src.cols - 1 || i == 1 || j == 1 || i == src.rows - 2 || j == src.cols - 2)
			{
				//padding with 0
				grayLevelA[i][j] = 0.0;
			}
			else
			{
				for (int x = 0; x < 5; x++)
				{
					for (int y = 0; y < 5; y++)
					{
						grayLevelA[i][j] = grayLevelA[i][j] + src.at<uchar>(i - 2 + x, j - 2 + y)* averagingMask[x][y];
					}
				}
			}
		}
	}
	
	//assign grayLevelA to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = grayLevelA[i][j];
		}
	}
	
}

void Sobel(const Mat &src, Mat &dst)
{

	float maskx[3][3] = { -1,-2,-1,0,0,0,1,2,1 };
	float masky[3][3] = { -1,0,1,-2,0,2,-1,0,1 };
	float min = 1000;
	float max = -1000;

	//|gx|
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (i == 0 || j == 0 || i == src.rows - 1 || j == src.cols - 1)
			{
				grayLevelx[i][j] = 0.0;

			}
			else
			{
				grayLevelx[i][j] = src.at<uchar>(i - 1, j)*maskx[0][1] + src.at<uchar>() + src.at<uchar>(i, j - 1)*maskx[1][0] + src.at<uchar>(i, j + 1)*maskx[1][2] + src.at<uchar>(i + 1, j)*maskx[2][1] + src.at<uchar>(i, j)*maskx[1][1] + src.at<uchar>(i - 1, j - 1)*maskx[0][0] + src.at<uchar>(i + 1, j - 1)*maskx[2][0] + src.at<uchar>(i - 1, j + 1)*maskx[0][2] + src.at<uchar>(i + 1, j + 1)*maskx[2][2];
			}

			grayLevelx[i][j] = abs(grayLevelx[i][j]);
		}
	}

	//|gy|
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (i == 0 || j == 0 || i == src.rows - 1 || j == src.cols - 1)
			{
				grayLevely[i][j] = 0.0;

			}
			else
			{
				grayLevely[i][j] = src.at<uchar>(i - 1, j)*masky[0][1] + src.at<uchar>() + src.at<uchar>(i, j - 1)*masky[1][0] + src.at<uchar>(i, j + 1)*masky[1][2] + src.at<uchar>(i + 1, j)*masky[2][1] + src.at<uchar>(i, j)*masky[1][1] + src.at<uchar>(i - 1, j - 1)*masky[0][0] + src.at<uchar>(i + 1, j - 1)*masky[2][0] + src.at<uchar>(i - 1, j + 1)*masky[0][2] + src.at<uchar>(i + 1, j + 1)*masky[2][2];
			}

			grayLevely[i][j] = abs(grayLevely[i][j]);
		}
	}

	//|gx|+|gy|
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevelx[i][j] = grayLevelx[i][j] + grayLevely[i][j];

			if (grayLevelx[i][j] < min)
			{
				min = grayLevelx[i][j];
			}
		}
	}

	//assign |gx|+|gy| to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = grayLevelx[i][j];
		}
	}
	/*
	//fm
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevelx[i][j] = grayLevelx[i][j] - min;

			if (grayLevelx[i][j] > max)
			{
				max = grayLevelx[i][j];
			}
		}
	}

	//assign fs to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 * (grayLevelx[i][j] / max);
		}
	}
	*/
}

void LaplacianMultiplyAveraging(Mat &src, Mat &src2, Mat &dst)
{
	float min = 1000000;
	float max = -1000000;

	//Lap*Ave
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = src.at<uchar>(i, j)*src2.at<uchar>(i, j);

			if (grayLevel[i][j] < min)
			{
				min = grayLevel[i][j];
			}
		}
	}

	//fm
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = grayLevel[i][j] - min;

			if (grayLevel[i][j] > max)
			{
				max = grayLevel[i][j];
			}
		}
	}

	//assign fs to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 * (grayLevel[i][j] / max);
		}
	}
	
}

void SrcPlusF(Mat &src,Mat &src2,Mat &dst)
{

	float min = 100000;
	float max = -100000;

	//SrcPlusF
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = src.at<uchar>(i, j) + src2.at<uchar>(i, j);

			if (grayLevel[i][j] < min)
			{
				min = grayLevel[i][j];
			}
		}
	}

	//fm
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			grayLevel[i][j] = grayLevel[i][j] - min;

			if (grayLevel[i][j] > max)
			{
				max = grayLevel[i][j];
			}
		}
	}

	//assign fs to dst
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = 255 * (grayLevel[i][j] / max);
		}
	}
}

void GammaTransform(Mat &src, Mat &dst)
{
	//GammaTransform
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j) = sqrt((src.at<uchar>(i, j))*0.00392) * 255;
		}
	}
}

int main(){

	// Read input images
	// Fig3.tif is in openCV\bin\Release
	Mat SrcImg = imread("Fig0343(a)(skeleton_orig).tif", CV_LOAD_IMAGE_GRAYSCALE);

	// Create a grayscale output image matrix
	Mat LaplacianDstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	Mat SrcImgPlusLaplacianDstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	Mat SobelDstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	Mat AveragingDstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	Mat LapXAve = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	Mat SrcPF = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);
	Mat GammaDstImg = Mat(SrcImg.rows, SrcImg.cols, CV_8UC1);

	// Copy each pixel of the source image to the output image
	
	Laplacian(SrcImg, LaplacianDstImg);
	SrcPlusLaplacian(SrcImg, SrcImgPlusLaplacianDstImg, LaplacianDstImg);
	Sobel(SrcImg, SobelDstImg);
	Averaging(SobelDstImg, AveragingDstImg);
	LaplacianMultiplyAveraging(SrcImgPlusLaplacianDstImg, AveragingDstImg, LapXAve);
	SrcPlusF(SrcImg, LapXAve, SrcPF);
	GammaTransform(SrcPF, GammaDstImg);
	
	// Show images
	imshow("Input Image (a)", SrcImg);
	imshow("Output Image (b)", LaplacianDstImg);
	imshow("Output Image (c)", SrcImgPlusLaplacianDstImg);
	imshow("Output Image (d)", SobelDstImg);
	imshow("Output Image (e)", AveragingDstImg);
	imshow("Output Image (f)", LapXAve);
	imshow("Output Image (g)", SrcPF);
	imshow("Output Image (h)", GammaDstImg);
	
	//Write output images
	imwrite("(b).tif", LaplacianDstImg);
	imwrite("(c).tif", SrcImgPlusLaplacianDstImg);
	imwrite("(d).tif", SobelDstImg);
	imwrite("(e).tif", AveragingDstImg);
	imwrite("(f).tif", LapXAve);
	imwrite("(g).tif", SrcPF);
	imwrite("(h).tif", GammaDstImg);
	
	waitKey(0);
	return 0;
}