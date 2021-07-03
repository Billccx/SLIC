#include <iostream>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <queue>
#include <map>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

double step8nbr[8][2] = { {-1,-1},{0,-1},{1,-1},{1,0},{1,1},{0,1},{-1,1},{-1,0} };
double step4nbr[4][2] = { {-1,0},{0,1},{1,0},{0,-1} };
//初始间距
int S;

//初始超像素个数
const int nsp = 256;

//距离权重
const double m = 40;

//迭代次数
const int epoch = 10;

//记录当前像素到聚类中心的距离
vector<vector<double>> dis;

//记录当前像素对应的聚类中心
vector<vector<int>> cluster;

//记录所有聚类中心所包含的像素个数
vector<int> clustercnt;

//聚类中心
typedef struct node {
	int x, y;
	double l, a, b;
}point;
vector<point> centers;

//xyz颜色空间像素
typedef struct xyzcolor {
	float x, y, z;
	xyzcolor(float a, float b, float c) {
		x = a;
		y = b;
		z = c;
	}
}xyzColor;

//lab颜色空间像素
typedef struct labcolor {
	float l, a, b;
	labcolor(float x, float y, float z) {
		l = x;
		a = y;
		b = z;
	}
}labColor;

//rgb颜色空间像素
typedef struct rgbcolor {
	float r, g, b;
	rgbcolor(float x, float y, float z) {
		r = x;
		g = y;
		b = z;
	}
}rgbColor;


//rgb转xyz
xyzColor rgbToXyz(rgbColor c) {
	float x, y, z, r, g, b;

	r = c.r / 255.0; g = c.g / 255.0; b = c.b / 255.0;

	if (r > 0.04045)
		r = powf(((r + 0.055) / 1.055), 2.4);
	else r /= 12.92;

	if (g > 0.04045)
		g = powf(((g + 0.055) / 1.055), 2.4);
	else g /= 12.92;

	if (b > 0.04045)
		b = powf(((b + 0.055) / 1.055), 2.4);
	else b /= 12.92;

	r *= 100; g *= 100; b *= 100;

	x = r * 0.4124 + g * 0.3576 + b * 0.1805;
	y = r * 0.2126 + g * 0.7152 + b * 0.0722;
	z = r * 0.0193 + g * 0.1192 + b * 0.9505;

	return xyzColor(x, y, z);
}

//xyz转lab
labColor xyzToCIELAB(xyzColor c) {
	float x, y, z, l, a, b;
	const float refX = 95.047, refY = 100.0, refZ = 108.883;

	x = c.x / refX; y = c.y / refY; z = c.z / refZ;

	if (x > 0.008856)
		x = powf(x, 1 / 3.0);
	else x = (7.787 * x) + (16.0 / 116.0);

	if (y > 0.008856)
		y = powf(y, 1 / 3.0);
	else y = (7.787 * y) + (16.0 / 116.0);

	if (z > 0.008856)
		z = powf(z, 1 / 3.0);
	else z = (7.787 * z) + (16.0 / 116.0);

	l = 116 * y - 16;
	a = 500 * (x - y);
	b = 200 * (y - z);

	return labColor(l, a, b);
}

//获取3*3区域中的最小梯度
void GetMinGrad(point& p, Mat img) {
	double mingrad = FLT_MAX;
	for (int i = p.y - 1; i <= p.y + 1; i++) {
		for (int j = p.x - 1; j <= p.x + 1; j++) {
			int right;
		}
	}
}

//计算梯度
void CalGrad(Mat grad, Mat intense) {
	for (int i = 0; i < grad.rows; i++) {
		for (int j = 0; j < grad.cols; j++) {
			double right, down, now = (double)intense.at<uchar>(i, j);
			if (j + 1 < grad.cols) right = (double)intense.at<uchar>(i, j + 1);
			else right = 0;
			if (i + 1 < grad.rows) down = (double)intense.at<uchar>(i + 1, j);
			else down = 0;
			grad.at<uchar>(i, j) = sqrt(pow(right - now, 2) + pow(down - now, 2));
		}
	}
}

//初始化
void InitImage(vector<vector<labColor>>& img, Mat grad) {
	//cols x     rows y
	int rows = img.size();
	int cols = img[0].size();

	for (int i = 0; i < rows; i++) {
		vector<int> clstrR;
		vector<double> disR;
		for (int j = 0; j < cols; j++) {
			clstrR.push_back(-1);
			disR.push_back(FLT_MAX);
		}
		dis.push_back(disR);
		cluster.push_back(clstrR);
	}


	//在邻域内寻找最小梯度点
	for (int i = S / 2; i < rows; i += S) {
		for (int j = S / 2; j < cols; j += S) {

			point temp;
			double mingrad = FLT_MAX;
			for (int k = i - 1; k <= i + 1; k++) {
				for (int l = j - 1; l <= j + 1; l++) {
					if (k < 0 || k >= rows || l < 0 || l >= cols) continue;
					double now = (double)grad.at<uchar>(k, l);
					if (now < mingrad) {
						mingrad = now;
						temp.x = l;
						temp.y = k;
					}
				}
			}

			temp.l = img[temp.y][temp.x].l;
			temp.a = img[temp.y][temp.x].a;
			temp.b = img[temp.y][temp.x].b;

			centers.push_back(temp);//插入一个聚类中心
			clustercnt.push_back(0);//将该聚类中心包含的像素初始化为0
			//if (temp.x != i || temp.y != j) cnt++;
		}
	}
}


//计算像素间距离
double CalDistence(point x, point y) {
	double dc = sqrt(pow(x.l - y.l, 2) + pow(x.a - y.a, 2) + pow(x.b - y.b, 2));
	double ds = sqrt(pow(x.x - y.x, 2) + pow(x.y - y.y, 2));
	double D = sqrt(pow(dc, 2) + pow(ds / S, 2) * pow(m, 2));
	return D;
}


//计算迭代残差
double CalResError(vector<point>& pre, vector<point>& now) {
	double re = 0;
	for (int i = 0; i < pre.size(); i++) {
		re += sqrt(pow(pre[i].x - now[i].x, 2) + pow(pre[i].y - now[i].y, 2));
	}
	return re;
}


//生成超像素
void GenerateSuperpixel(vector<vector<labColor>>& img, int S) {
	int rows = img.size();
	int cols = img[0].size();

	//迭代次数
	for (int iter = 0; iter < epoch; iter++) {
		//遍历所有聚类中心
		for (int t = 0; t < centers.size(); t++) {
			point center = centers[t];
			//2S*2S中的行
			for (int i = center.y - S; i <= center.y + S; i++) {
				//2S*2S中的列
				for (int j = center.x - S; j <= center.x + S; j++) {
					if (i >= 0 && i < rows && j >= 0 && j < cols) {
						point dst;
						dst.x = j;
						dst.y = i;
						dst.l = img[i][j].l;
						dst.a = img[i][j].a;
						dst.b = img[i][j].b;
						double D = CalDistence(center, dst);
						if (D < dis[i][j]) {
							dis[i][j] = D;
							cluster[i][j] = t;
						}
					}
				}
			}
		}

		vector<point> pre = centers;

		for (int i = 0; i < centers.size(); i++) {
			centers[i].l = centers[i].a = centers[i].b = centers[i].x = centers[i].y = 0;
		}


		//计算新聚类中心
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				int index = cluster[i][j];
				//cout << index << endl;
				if (index < 0) {
					cout << "error" << endl;
					cout << "row=" << i << " cols=" << j << endl;
					continue;
				}
				centers[index].l += img[i][j].l;
				centers[index].a += img[i][j].a;
				centers[index].b += img[i][j].b;
				centers[index].x += j;
				centers[index].y += i;
				clustercnt[index]++;
			}
		}

		//正则化
		for (int i = 0; i < centers.size(); i++) {
			centers[i].l /= clustercnt[i];
			centers[i].a /= clustercnt[i];
			centers[i].b /= clustercnt[i];
			centers[i].x /= clustercnt[i];
			centers[i].y /= clustercnt[i];
		}


		double residualerror = CalResError(pre, centers);
		printf("epoch=%d,error=%lf\n", iter, residualerror);

	}
}

//绘制超像素边界
void DrawSuperPixel(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			bool isEdge = false;
			for (int k = 0; k < 4; k++) {
				int y = i + step4nbr[k][0];
				int x = j + step4nbr[k][1];
				if (y >= 0 && y < img.rows && x >= 0 && x < img.cols
					&& cluster[i][j] != cluster[y][x]) {
					isEdge = true;
					break;
				}
			}
			if (isEdge) {
				Point p(j, i);
				circle(img, p, 0, Scalar(0, 0, 0), -1);
			}
		}
	}
}


//强制连续性
void EnforceConnectivity(vector<vector<labColor>>& img) {

	int rows = img.size();
	int cols = img[0].size();

	int adjlabel = 0, label = 0;

	int threshold = rows * cols / centers.size();
	vector<vector<int>> newcluster;

	for (int i = 0; i < rows; i++) {
		vector<int> newrows;
		for (int j = 0; j < cols; j++) {
			newrows.push_back(-1);
		}
		newcluster.push_back(newrows);
	}

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			if (newcluster[i][j] == -1) {
				newcluster[i][j] = label;
				//BFS
				queue<pair<int, int>> q;   //BFS队列
				vector<pair<int, int>> connectiveregion;   //连通区域

				q.push(pair<int, int>(i, j));
				connectiveregion.push_back(pair<int, int>(i, j));

				//在四邻域中寻找新聚类中心
				for (int t = 0; t < 4; t++) {
					int y = i + step4nbr[t][0];
					int x = j + step4nbr[t][1];
					if (y >= 0 && y < rows && x >= 0 && x < cols) {
						if (newcluster[y][x] >= 0) {
							adjlabel = newcluster[y][x];
						}
					}
				}

				int numclusterpixel = 1;
				while (!q.empty()) {
					pair<int, int> now = q.front();
					q.pop();
					for (int k = 0; k < 4; k++) {
						int y = now.first + step4nbr[k][0];
						int x = now.second + step4nbr[k][1];
						if (y >= 0 && y < rows && x >= 0 && x < cols
							&& newcluster[y][x] == -1 && cluster[y][x] == cluster[i][j]) {
							numclusterpixel++;
							q.push(pair<int, int>(y, x));
							connectiveregion.push_back(pair<int, int>(y, x));
							newcluster[y][x] = label;
						}
					}
				}

				//cout << connectiveregion.size() << endl;
				//cout << numclusterpixel << endl;
				//区域面积小于阈值，进行合并
				if (numclusterpixel <= threshold / 2) {
					for (int k = 0; k < connectiveregion.size(); k++) {
						newcluster[connectiveregion[k].first][connectiveregion[k].second] = adjlabel;
					}
					//cout << adjlabel << endl;
					label--;
				}
				label++;
			}
		}
	}

	/*
	for (int i = 110; i < 150; i++) {
		for (int j = 280; j < 310; j++) {
			printf("%3d ", cluster[i][j]);
		}
		cout << endl;
	}
	*/
	cluster = newcluster;
	//cout << "________________________________________________" << endl;
}


typedef struct colorinf {
	double r, g, b;
}RGBInfo;

typedef struct sprpxso {
	double r, g, b;
	int cnt, x, y, label;
}SuperPixel;

vector<SuperPixel> SuperPixels;
vector<SuperPixel> KMcenters;
bool cmp(const SuperPixel& a, const SuperPixel& b) {
	return a.label < b.label;
}


//更新超像素中心信息
void UpdateCenters(Mat img) {
	for (int i = 0; i < cluster.size(); i++) {
		for (int j = 0; j < cluster[i].size(); j++) {
			bool isFind = false;
			for (int k = 0; k < SuperPixels.size(); k++) {
				if (SuperPixels[k].label == cluster[i][j]) {
					SuperPixels[k].b += img.at<cv::Vec3b>(i, j)[0];
					SuperPixels[k].g += img.at<cv::Vec3b>(i, j)[1];
					SuperPixels[k].r += img.at<cv::Vec3b>(i, j)[2];
					SuperPixels[k].x += j;
					SuperPixels[k].y += i;
					SuperPixels[k].cnt++;
					isFind = true;
					break;
				}
			}
			if (!isFind) {
				SuperPixel temp;
				temp.b = img.at<cv::Vec3b>(i, j)[0];
				temp.g = img.at<cv::Vec3b>(i, j)[1];
				temp.r = img.at<cv::Vec3b>(i, j)[2];
				temp.x = j;
				temp.y = i;
				temp.cnt = 1;
				temp.label = cluster[i][j];
				SuperPixels.push_back(temp);
			}
		}
	}

	for (int i = 0; i < SuperPixels.size(); i++) {
		SuperPixels[i].r /= SuperPixels[i].cnt;
		SuperPixels[i].g /= SuperPixels[i].cnt;
		SuperPixels[i].b /= SuperPixels[i].cnt;
	}
	sort(SuperPixels.begin(), SuperPixels.end(), cmp);
}

//将每个超像素中包含的像素RGB替换为平均值
void ReplacePixelColour(Mat img) {
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<cv::Vec3b>(i, j)[0] = (uchar)SuperPixels[cluster[i][j]].b;
			img.at<cv::Vec3b>(i, j)[1] = (uchar)SuperPixels[cluster[i][j]].g;
			img.at<cv::Vec3b>(i, j)[2] = (uchar)SuperPixels[cluster[i][j]].r;
		}
	}
}


int main() {
	//2.jpg   dog.png   p1.jpg

	string path = "word.png";
	Mat src = imread(path);

	int N = src.cols * src.rows;
	S = (int)sqrt(N / nsp);
	//cout << "S=" << S << endl;

	Mat intense = imread(path, 0);
	Mat grad = Mat::zeros(intense.size(), intense.type());
	CalGrad(grad, intense);

	//Mat cielab = Mat::zeros(intense.size(), intense.type());
	//cvtColor(src, cielab, COLOR_BGR2Lab);

	vector<vector<labColor>> cielab;
	for (int i = 0; i < src.rows; i++) {
		vector<labColor> row;
		for (int j = 0; j < src.cols; j++) {
			row.push_back(labColor(0, 0, 0));
		}
		cielab.push_back(row);
	}



	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			float xr, xg, xb, yl, ya, yb;
			xr = src.at<cv::Vec3b>(i, j)[2];
			xg = src.at<cv::Vec3b>(i, j)[1];
			xb = src.at<cv::Vec3b>(i, j)[0];

			xyzColor temp = rgbToXyz(rgbColor(xr, xg, xb));
			labColor temp2 = xyzToCIELAB(temp);

			cielab[i][j].l = temp2.l;
			cielab[i][j].a = temp2.a;
			cielab[i][j].b = temp2.b;
		}
	}


	/*
	Mat intense = imread("dog.png", 0);
	Mat grad = Mat::zeros(intense.size(), intense.type());
	CalGrad(grad, intense);
	*/

	//cout << src << endl;
	InitImage(cielab, grad);


	/*
	for (int i = 0; i < centers.size(); i++) {
		printf("x=%d,y=%d\n", centers[i].x, centers[i].y);
		Point p(centers[i].x, centers[i].y);
		circle(src, p, 0, Scalar(0, 255, 0), -1);
	}
	*/

	GenerateSuperpixel(cielab, S);
	for (int i = 0; i < 1; i++) {
		EnforceConnectivity(cielab);
		//cout << i << endl;
	}

	DrawSuperPixel(src);

	imwrite("SuperPixels.jpg", src);


	Mat src2 = imread(path);
	UpdateCenters(src2);
	ReplacePixelColour(src2);
	imwrite("SuperPixelSegment.jpg", src2);

	//namedWindow("Display", CV_WINDOW_AUTOSIZE);
	//imshow("Display", src);
	//waitKey(0);
}