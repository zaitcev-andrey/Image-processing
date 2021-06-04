#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <ctime>

#define PI 3.14159265

using namespace std;
using namespace cv;

int clamp(int value, int max, int min)
{
	value = value > max ? max : value;
	return value < min ? min : value;
}

void Gray_Filter(const Mat& photo, Mat& gray_photo)
{
	for (int i = 0; i < photo.rows; i++)
	{
		for (int j = 0; j < photo.cols; j++)
		{
			float b = photo.at<Vec3b>(i, j)[0] * 0.11;
			float g = photo.at<Vec3b>(i, j)[1] * 0.59;
			float r = photo.at<Vec3b>(i, j)[2] * 0.3;

			float gray = b + g + r;
			Vec3f vec = { gray, gray, gray };
			gray_photo.at<Vec3b>(i, j) = vec;
		}
	}
}

vector<float> createGaussianKernel(int radius, int sigma)
{
	const unsigned int diametr = 2 * radius + 1;
	float norm = 0; // ����������� ���������� ����
	vector<float> kernel(diametr * diametr); // ������ ���� �������

	// ������������ ���� ��������� �������
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++)
		{
			int idx = (i + radius) * diametr + j + radius;
			kernel[idx] = exp(-(i * i + j * j) / (sigma * sigma));
			norm += kernel[idx];
		}
	}

	// ��������� ����
	for (int i = 0; i < diametr * diametr; i++)
		kernel[i] /= norm;
	return kernel;
}

/*void gauss_filter_color(const Mat& source_photo, Mat& new_photo, int radius, int sigma)
{
	const unsigned int diametr = 2 * radius + 1;
	vector<float> kernel = createGaussianKernel(radius, sigma);
	for (int x = 0; x < source_photo.rows; x++)
	{
		for (int y = 0; y < source_photo.cols; y++)
		{
			float b = 0;
			float g = 0;
			float r = 0;
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					int idx = (i + radius) * diametr + j + radius;

					Vec3b color = source_photo.at<Vec3b>(clamp(x + j, source_photo.rows - 1, 0), clamp(y + i, source_photo.cols - 1, 0));

					b += color[0] * kernel[idx];
					g += color[1] * kernel[idx];
					r += color[2] * kernel[idx];
				}
			}
			Vec3i v = { clamp(b, 255, 0), clamp(g, 255, 0), clamp(r, 255, 0) };
			new_photo.at<Vec3b>(x, y) = v;
		}
	}
}*/

void gauss_filter_gray(const Mat& source_photo, Mat& new_photo, int radius, int sigma) // ��� ��������� �����
{
	const unsigned int diametr = 2 * radius + 1;
	vector<float> kernel = createGaussianKernel(radius, sigma);
	for (int x = 0; x < source_photo.rows; x++)
	{
		for (int y = 0; y < source_photo.cols; y++)
		{
			float value = 0;
			for (int i = -radius; i <= radius; i++)
			{
				for (int j = -radius; j <= radius; j++)
				{
					int idx = (i + radius) * diametr + j + radius;

					unsigned char color = source_photo.at<Vec3b>(clamp(x + j, source_photo.rows - 1, 0), clamp(y + i, source_photo.cols - 1, 0))[0];

					value += color * kernel[idx];
				}
			}
			Vec3i v = { clamp(value, 255, 0), clamp(value, 255, 0), clamp(value, 255, 0) };
			new_photo.at<Vec3b>(x, y) = v;
		}
	}
}

/*void sobel_operator_color(const Mat& source_photo, Mat& new_photo)
{
	int diametr = 3; // 3, ��� ��� ������ ������ ����� 1
	vector<int> kernel(diametr * diametr);

	kernel[0] = -1;
	kernel[1] = 0;
	kernel[2] = 1;
	kernel[3] = -2;
	kernel[4] = 0;
	kernel[5] = 2;
	kernel[6] = -1;
	kernel[7] = 0;
	kernel[8] = 1;

	for (int x = 0; x < source_photo.rows; x++)
	{
		for (int y = 0; y < source_photo.cols; y++)
		{
			int b = 0, g = 0, r = 0;
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					int idx = (i + 1) * diametr + j + 1;

					Vec3b color = source_photo.at<Vec3b>(clamp(x + j, source_photo.rows - 1, 0), clamp(y + i, source_photo.cols - 1, 0));

					b += color[0] * kernel[idx];
					g += color[1] * kernel[idx];
					r += color[2] * kernel[idx];
				}
			}
			Vec3i v = { clamp(b, 255, 0), clamp(g, 255, 0), clamp(r, 255, 0) };
			new_photo.at<Vec3b>(x, y) = v;
		}
	}
}*/

void sobel_operator_gray(const Mat& source_photo, Mat& new_photo_Gx, Mat& new_photo_Gy) // ��� ��������� �����
{
	int diametr = 3; // 3, ��� ��� ������ ������ ����� 1
	vector<int> kernel_Gx(diametr * diametr);
	vector<int> kernel_Gy(diametr * diametr);
	
	// ���� ��� ������������� �����������
	kernel_Gx[0] = 1;
	kernel_Gx[1] = 2;
	kernel_Gx[2] = 1;
	kernel_Gx[3] = 0;
	kernel_Gx[4] = 0;
	kernel_Gx[5] = 0;
	kernel_Gx[6] = -1;
	kernel_Gx[7] = -2;
	kernel_Gx[8] = -1;

	// ���� ��� ��������������� �����������
	kernel_Gy[0] = 1;
	kernel_Gy[1] = 0;
	kernel_Gy[2] = -1;
	kernel_Gy[3] = 2;
	kernel_Gy[4] = 0;
	kernel_Gy[5] = -2;
	kernel_Gy[6] = 1;
	kernel_Gy[7] = 0;
	kernel_Gy[8] = -1;

	for (int x = 0; x < source_photo.rows; x++)
	{
		for (int y = 0; y < source_photo.cols; y++)
		{
			int value_Gx = 0;
			int value_Gy = 0;
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					int idx = (i + 1) * diametr + j + 1;

					unsigned char color = source_photo.at<Vec3b>(clamp(x + j, source_photo.rows - 1, 0), clamp(y + i, source_photo.cols - 1, 0))[0];

					value_Gx += color * kernel_Gx[idx];
					value_Gy += color * kernel_Gy[idx];
				}
			}
			Vec3i v_Gx = { clamp(value_Gx, 255, 0), clamp(value_Gx, 255, 0), clamp(value_Gx, 255, 0) };
			new_photo_Gx.at<Vec3b>(x, y) = v_Gx;
			Vec3i v_Gy = { clamp(value_Gy, 255, 0), clamp(value_Gy, 255, 0), clamp(value_Gy, 255, 0) };
			new_photo_Gy.at<Vec3b>(x, y) = v_Gy;
		}
	}
}

void suppression_of_non_maxima(const Mat& sobel_Gx, const Mat& sobel_Gy, const Mat& gauss_photo, Mat& new_photo)
{
	for (int x = 0; x < gauss_photo.rows; x++)
	{
		for (int y = 0; y < gauss_photo.cols; y++)
		{
			int Gx = sobel_Gx.at<Vec3b>(x, y)[0];
			int Gy = sobel_Gy.at<Vec3b>(x, y)[0];
			if (Gy == 0)
			{
				Gy = 1;
				//new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
				//continue;
			}
			int Q = atan(Gx / Gy) * 180 / PI; // ������� ����(� ��������), �� �������� ����� �������� ��������� �������� �� ������-�� �� �����������

			if (Q > 180)
				Q -= 180;

			if (0 <= Q && Q <= 45)
			{
				if (Q - 0 < 45 - Q)
					Q = 0;
				else
					Q = 45;
			}
			else if (45 <= Q && Q <= 90)
			{
				if (Q - 45 < 90 - Q)
					Q = 45;
				else
					Q = 90;
			}
			else if (90 <= Q && Q <= 135)
			{
				if (Q - 90 < 135 - Q)
					Q = 90;
				else
					Q = 135;
			}
			else if (135 <= Q && Q <= 180)
			{
				if (Q - 135 < 180 - Q)
					Q = 135;
				else
					Q = 180;
			}

			switch (Q)
			{
			case 0:
				if (gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x - 1, gauss_photo.rows - 1, 0), y)[0]
					|| gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x + 1, gauss_photo.rows - 1, 0), y)[0])
				{
					new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
				}
				break;
			case 180:
				if (gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x - 1, gauss_photo.rows - 1, 0), y)[0]
					|| gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x + 1, gauss_photo.rows - 1, 0), y)[0])
				{
					new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
				}
				break;
			case 45:
				if (gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x - 1, gauss_photo.rows - 1, 0), clamp(y + 1, gauss_photo.cols - 1, 0))[0]
					|| gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x + 1, gauss_photo.rows - 1, 0), clamp(y - 1, gauss_photo.cols - 1, 0))[0])
				{
					new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
				}
				break;
			case 135:
				if (gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x - 1, gauss_photo.rows - 1, 0), clamp(y - 1, gauss_photo.cols - 1, 0))[0]
					|| gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(clamp(x + 1, gauss_photo.rows - 1, 0), clamp(y + 1, gauss_photo.cols - 1, 0))[0])
				{
					new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
				}
				break;
			case 90:
				if (gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(x, clamp(y - 1, gauss_photo.cols - 1, 0))[0]
					|| gauss_photo.at<Vec3b>(x, y)[0] <= gauss_photo.at<Vec3b>(x, clamp(y + 1, gauss_photo.cols - 1, 0))[0])
				{
					new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
				}
				break;
			}
		}
	}
}

void double_threshold_filtering(const Mat& source_photo, Mat& new_photo, int high_pr, int low_pr)
{
	for (int x = 0; x < source_photo.rows; x++)
	{
		for (int y = 0; y < source_photo.cols; y++)
		{
			if (source_photo.at<Vec3b>(x, y)[0] >= high_pr)
				new_photo.at<Vec3b>(x, y) = { 255, 255, 255 };
			else if(source_photo.at<Vec3b>(x, y)[0] <= low_pr)
				new_photo.at<Vec3b>(x, y) = { 0, 0, 0 };
			else
				new_photo.at<Vec3b>(x, y) = { 127, 127, 127 };
		}
	}
}

int main()
{
	// �������� ��������� ������ �����
	Mat source_photo = imread("Kat.jpg");

	// ����� ����������� ��������� �����, ����������� ����������� � ������� ������, ����� ��������� �������������� �������.

	// 1) ����������� � ������� ������
	Mat gray_photo(source_photo.size(), source_photo.type());
	Gray_Filter(source_photo, gray_photo);

	imwrite("gray_photo.jpg", gray_photo);

	// 2) ��������� ������ ������ ��� ���������� �� ����
	Mat photo_after_gauss_filter(gray_photo.size(), gray_photo.type());
	gauss_filter_gray(gray_photo, photo_after_gauss_filter, 1, 7); // ��� ������ �����, ��� ������ ���� �� ���� �������� �������� ����

	imwrite("gray_photo_after_gauss_filter.jpg", photo_after_gauss_filter);

	// 3) ����� ����������(���������� �������� ������). ������� ����� ���, ��� �������� ����������� ��������� ������������ ��������
	Mat photo_after_sobel_Gx(photo_after_gauss_filter.size(), photo_after_gauss_filter.type());
	Mat photo_after_sobel_Gy(photo_after_gauss_filter.size(), photo_after_gauss_filter.type());
	sobel_operator_gray(photo_after_gauss_filter, photo_after_sobel_Gx, photo_after_sobel_Gy);

	imwrite("gray_photo_after_sobel_Gx.jpg", photo_after_sobel_Gx);
	imwrite("gray_photo_after_sobel_Gy.jpg", photo_after_sobel_Gy);

	// 4) ���������� ������������
	Mat photo_after_suppression(photo_after_gauss_filter.size(), photo_after_gauss_filter.type());
	suppression_of_non_maxima(photo_after_sobel_Gx, photo_after_sobel_Gy, photo_after_gauss_filter, photo_after_suppression);

	imwrite("photo_after_suppression.jpg", photo_after_suppression);

	// 5) ������� ��������� ����������
	Mat Kenny_photo(photo_after_suppression.size(), photo_after_suppression.type());
	double_threshold_filtering(photo_after_suppression, Kenny_photo, 180, 74);

	imwrite("photo_after_Kenny_algorithm.jpg", Kenny_photo);

	system("pause");
	waitKey();
}