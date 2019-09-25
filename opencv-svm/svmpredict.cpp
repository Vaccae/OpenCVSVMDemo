#include <iostream>
#include <opencv2/opencv.hpp>

std::string trainfile = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/hog_train.xml";
std::string videofile = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/vedio/2.mp4";

void hog_deal(cv::Mat& src, std::vector<float>& dst)
{
	//定义Hog，输入图像必须能被2整除
	cv::HOGDescriptor hog;
	//获取默认的值 64 * 128
	int imgwidth = hog.winSize.width;
	int imgheight = hog.winSize.height;

	int h = src.rows;
	int w = src.cols;
	float rate = (float)imgwidth / w;

	cv::Mat tmpsrc, gray;
	//变换图像为64 * 128的大小
	cv::resize(src, tmpsrc, cv::Size(imgwidth, int(rate*h)));
	//灰度转换
	cv::cvtColor(tmpsrc, gray, cv::COLOR_BGR2GRAY);

	//为了保证源图的比例，如果图片大于64*128，就截取roi区域
	//如果小于64*128就填充，保证图片信息不会丢失
	cv::Mat result = cv::Mat::zeros(cv::Size(imgwidth, imgheight), CV_8UC1);
	//新建Mat填充127灰度
	result = cvScalar(127);
	cv::Rect roi;
	roi.x = 0;
	roi.width = imgwidth;
	roi.y = (imgheight - gray.rows) / 2;
	roi.height = gray.rows;

	gray.copyTo(result(roi));
	//计算描述子，一般窗口步长都用Size(8,8)就是8*8
	hog.compute(result, dst, cv::Size(8, 8), cv::Size(0, 0));
}

int main(int argc, char** argv)
{
	cv::Mat frame;
	cv::Mat gray;

	cv::Mat tmpsrc;

	//加载训练文件
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(trainfile);

	try
	{
		//读取视频文件
		cv::VideoCapture video;
		video.open(videofile);
		if (!video.isOpened())
		{
			printf("could not read video...");
			getchar();
			return -1;
		}

		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

		int rectcount = 0;
		int predict = 0;
		//读取图像每一帧
		while (video.read(frame))
		{
			//对图像进行缩放
			cv::resize(frame, frame, cv::Size(0, 0), 0.8, 0.8);
			//高斯模糊
			cv::GaussianBlur(frame, gray, cv::Size(3, 3), 0.5, 0.5);
			//转为灰度图
			cv::cvtColor(gray, gray, CV_BGR2GRAY);
			//图像二值化操作
			cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			//形态学闭操作
			cv::morphologyEx(gray, gray, CV_MOP_CLOSE, kernel);

			//寻找轮廓
			std::vector<std::vector<cv::Point>> contours;
		    //hierarchy:可选的输出向量(std::vector)，包含了图像的拓扑信息，
			//作为轮廓数量的表示hierarchy包含了很多元素，
			//每个轮廓contours[i]对应hierarchy中hierarchy[i][0]~hierarchy[i][3], 
			//分别表示后一个轮廓，前一个轮廓，父轮廓，内嵌轮廓的索引，
			//如果没有对应项，则相应的hierarchy[i]设置为负数。
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(gray, contours, hierarchy, CV_RETR_CCOMP, 
				CV_CHAIN_APPROX_SIMPLE);
			//遍历轮廓排除不可能的轮廓再分析
			for (size_t t = 0; t < contours.size(); t++) {
				//寻找最小矩形
				cv::RotatedRect currentrect = cv::minAreaRect(contours[t]);
				cv::Rect rect = currentrect.boundingRect();

				//判断有没有父轮廓，如果有，直接跳过，我们检测父轮廓
				if(hierarchy[t][2]>0) continue;

				//判断轮廓在图像中大于五分之一，小于五份之四的轮廓为检测轮廓
				int width = frame.cols / 5 * 4;
				int height = frame.rows / 5 * 4;
				int width2 = frame.cols - width;
				int height2 = frame.rows - height;

				if (rect.width < width && rect.height < height
					&& rect.width>width2 && rect.width>height2
					&& rect.x > 0 && rect.y > 0
					&& rect.x + rect.width < frame.cols
					&& rect.y + rect.height < frame.rows)
				{
					rectcount++;//记录疑似需要预测的个数
					tmpsrc = frame(rect);
					std::vector<float> fv;

					try
					{
						//HOG处理
						hog_deal(tmpsrc, fv);
						cv::Mat one_row = cv::Mat::zeros(cv::Size(fv.size(), 1), CV_32FC1);
						for (int i = 0; i < fv.size(); i++) {
							one_row.at<float>(0, i) = fv[i];
						}
						//进行svm的预测
						float result = svm->predict(one_row);
						printf("result:%f\n", result);
						//当结果大于0说明预测匹配到了，画上矩形
						if (result > 0)
						{
							predict++;//记录匹配个数
							cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 3, 8, 0);
						}
					}
					catch (const std::exception& exception)
					{
						printf("predict error:%s\n", exception.what());
					}
				}
			}
			printf("rectcount:%d,predictcount:%d\n", rectcount, predict);
			cv::imshow("kindle detection", frame);
			char c = cv::waitKey(10);
			if (c == 27)
			{
				break;
			}
		}
		video.release();
		cv::waitKey(0);
		return 0;
	}
	catch (const std::exception& ex)
	{
		printf(ex.what());
		getchar();
		return -1;
	}

}