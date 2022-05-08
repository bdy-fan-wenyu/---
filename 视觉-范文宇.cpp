#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

VideoCapture cap(1);                                                                                     //打开USB摄像头

#define KNOWN_DISTANCE  30
#define KNOWN_WIDTH    10
#define KNOWN_HEIGHT   10
#define KNOWN_FOCAL_LENGTH  500 
double FocalLength = 0.0;                                                                                //定义测距相关变量
int main()
{
	RNG rng;
	//1.kalman filter setup  
	const int stateNum = 4;                                                                             //状态值4×1向量(x,y,△x,△y)  
	const int measureNum = 2;                                                                           //测量值2×1向量(x,y)    
	KalmanFilter KF(stateNum, measureNum, 0);                                                           //定义卡尔曼滤波器

	KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1);        //转移矩阵A  
	setIdentity(KF.measurementMatrix);                                                                  //测量矩阵H  
	setIdentity(KF.processNoiseCov, Scalar::all(1));                                                    //系统噪声方差矩阵Q  
	setIdentity(KF.measurementNoiseCov, Scalar::all(1e-60));                                            //测量噪声方差矩阵R  
	setIdentity(KF.errorCovPost, Scalar::all(10000));                                                   //后验错误估计协方差矩阵P    
	Mat measurement = Mat::zeros(measureNum, 1, CV_32F);                                                //初始测量值x'(0)，因为后面要更新这个值，所以必须先定义  



	while (1)
	{
		Mat SrcImage;                                                                                    
		cap >> SrcImage;                                                                                //逐帧抓取摄像头获取画面
		Mat orangeROI = SrcImage.clone();                                  
		Mat kernel = getStructuringElement(MORPH_RECT, Size(4, 4), Point(-1, -1));                      
		inRange(orangeROI, Scalar(20, 43, 46), Scalar(25, 255, 255), orangeROI);                        //分离橙色
		threshold(orangeROI, orangeROI, 240, 245, THRESH_BINARY);                                       //二值化
		blur(orangeROI, orangeROI, Size(4, 4));                                                         //模糊化
		dilate(orangeROI, orangeROI, kernel, Point(-1, -1), 8);                                         //膨胀
		vector<RotatedRect> vc;
		vector<RotatedRect> vRec;
		vector<vector<Point>>Light_Contour;
		findContours(orangeROI, Light_Contour, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);                     //找出橙色光的轮廓
		for (int i = 0; i < Light_Contour.size(); i++)
		{
			
			float Light_Contour_Area = contourArea(Light_Contour[i]);                                   //求轮廓面积        
			
			if (Light_Contour_Area < 15 || Light_Contour[i].size() <= 20)                               //去除较小轮廓fitllipse的限制条件
				continue;
			
			RotatedRect Light_Rec = fitEllipse(Light_Contour[i]);                                       // 用椭圆拟合区域得到外接矩形
			
					
			Light_Rec.size.height *= 1.1;
			Light_Rec.size.width *= 1.1;                                                                // 扩大灯柱的面积
			vc.push_back(Light_Rec);
		}

		
		for (size_t i = 0; i < vc.size(); i++)                                                          //从灯条长宽比上来筛选轮廓
		{
			for (size_t j = i + 1; (j < vc.size()); j++)
			{
				
				
				float Contour_Len1 = abs(vc[i].size.height - vc[j].size.height) / max(vc[i].size.height, vc[j].size.height);//长度差比率
				
				float Contour_Len2 = abs(vc[i].size.width - vc[j].size.width) / max(vc[i].size.width, vc[j].size.width);//宽度差比率
				


				RotatedRect ARMOR;                                                                        //定义装甲板的外接矩形
				ARMOR.center.x = (vc[i].center.x + vc[j].center.x) / 2.;                                  //x坐标
				ARMOR.center.y = (vc[i].center.y + vc[j].center.y) / 2.;                                  //y坐标
				ARMOR.angle = (vc[i].angle + vc[j].angle) / 2.;                                           //角度
				float nh, nw, yDiff, xDiff;
				nh = (vc[i].size.height + vc[j].size.height) / 2;                                         //高度
				
				nw = sqrt((vc[i].center.x - vc[j].center.x) * (vc[i].center.x - vc[j].center.x) + (vc[i].center.y - vc[j].center.y) * (vc[i].center.y - vc[j].center.y));// 宽度
				float ratio = nw / nh;                                                                    //匹配到的装甲板的长宽比
				xDiff = abs(vc[i].center.x - vc[j].center.x) / nh;                                        //x差比率
				yDiff = abs(vc[i].center.y - vc[j].center.y) / nh;                                        //y差比率
				if (ratio < 1.0 || ratio > 5.0 || xDiff < 0.5 || yDiff > 2.5)
					continue;
				ARMOR.size.height = nh;
				ARMOR.size.width = nw;
				vRec.push_back(ARMOR);
				Point2f point1;
				Point2f point2;
				point1.x = vc[i].center.x; point1.y = vc[i].center.y + 20;
				point2.x = vc[j].center.x; point2.y = vc[j].center.y - 20;
				int xmidnum = (point1.x + point2.x) / 2;
				int ymidnum = (point1.y + point2.y) / 2;

				  
				measurement.at<float>(0) = (float)ARMOR.center.x;
				measurement.at<float>(1) = (float)ARMOR.center.y;                                        //update measurement

				
				KF.correct(measurement);                                                                 //update  
				Mat prediction = KF.predict();
				Point predict_pt = Point(prediction.at<float>(0), prediction.at<float>(1));   //预测值(x',y')  
				FocalLength = (ARMOR.size.height * KNOWN_DISTANCE) / KNOWN_WIDTH;
				double DistanceInches = 0.0;
				DistanceInches = (KNOWN_WIDTH * FocalLength) / ARMOR.size.width;
				DistanceInches = DistanceInches * 2.54;
				putText(SrcImage, format("Distance(CM):%f", DistanceInches), Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 1, LINE_8);
				
				Scalar color(0, 255, 255);
				rectangle(SrcImage, point1, point2, color, 2);//将装甲板框起来
				circle(SrcImage, ARMOR.center, 10, color);//在装甲板中心画一个圆
				circle(SrcImage, predict_pt, 10, color);
			}
		}

		//播放成果
		namedWindow("SrcImage", 0);
		imshow("SrcImage", SrcImage);
		//imshow("SrcImage", orangeROI);
		waitKey(1);
	}
	return 0;
}




