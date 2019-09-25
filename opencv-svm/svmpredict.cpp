#include <iostream>
#include <opencv2/opencv.hpp>

std::string trainfile = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/hog_train.xml";
std::string videofile = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/vedio/2.mp4";

void hog_deal(cv::Mat& src, std::vector<float>& dst)
{
	//����Hog������ͼ������ܱ�2����
	cv::HOGDescriptor hog;
	//��ȡĬ�ϵ�ֵ 64 * 128
	int imgwidth = hog.winSize.width;
	int imgheight = hog.winSize.height;

	int h = src.rows;
	int w = src.cols;
	float rate = (float)imgwidth / w;

	cv::Mat tmpsrc, gray;
	//�任ͼ��Ϊ64 * 128�Ĵ�С
	cv::resize(src, tmpsrc, cv::Size(imgwidth, int(rate*h)));
	//�Ҷ�ת��
	cv::cvtColor(tmpsrc, gray, cv::COLOR_BGR2GRAY);

	//Ϊ�˱�֤Դͼ�ı��������ͼƬ����64*128���ͽ�ȡroi����
	//���С��64*128����䣬��֤ͼƬ��Ϣ���ᶪʧ
	cv::Mat result = cv::Mat::zeros(cv::Size(imgwidth, imgheight), CV_8UC1);
	//�½�Mat���127�Ҷ�
	result = cvScalar(127);
	cv::Rect roi;
	roi.x = 0;
	roi.width = imgwidth;
	roi.y = (imgheight - gray.rows) / 2;
	roi.height = gray.rows;

	gray.copyTo(result(roi));
	//���������ӣ�һ�㴰�ڲ�������Size(8,8)����8*8
	hog.compute(result, dst, cv::Size(8, 8), cv::Size(0, 0));
}

int main(int argc, char** argv)
{
	cv::Mat frame;
	cv::Mat gray;

	cv::Mat tmpsrc;

	//����ѵ���ļ�
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(trainfile);

	try
	{
		//��ȡ��Ƶ�ļ�
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
		//��ȡͼ��ÿһ֡
		while (video.read(frame))
		{
			//��ͼ���������
			cv::resize(frame, frame, cv::Size(0, 0), 0.8, 0.8);
			//��˹ģ��
			cv::GaussianBlur(frame, gray, cv::Size(3, 3), 0.5, 0.5);
			//תΪ�Ҷ�ͼ
			cv::cvtColor(gray, gray, CV_BGR2GRAY);
			//ͼ���ֵ������
			cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			//��̬ѧ�ղ���
			cv::morphologyEx(gray, gray, CV_MOP_CLOSE, kernel);

			//Ѱ������
			std::vector<std::vector<cv::Point>> contours;
		    //hierarchy:��ѡ���������(std::vector)��������ͼ���������Ϣ��
			//��Ϊ���������ı�ʾhierarchy�����˺ܶ�Ԫ�أ�
			//ÿ������contours[i]��Ӧhierarchy��hierarchy[i][0]~hierarchy[i][3], 
			//�ֱ��ʾ��һ��������ǰһ������������������Ƕ������������
			//���û�ж�Ӧ�����Ӧ��hierarchy[i]����Ϊ������
			std::vector<cv::Vec4i> hierarchy;
			cv::findContours(gray, contours, hierarchy, CV_RETR_CCOMP, 
				CV_CHAIN_APPROX_SIMPLE);
			//���������ų������ܵ������ٷ���
			for (size_t t = 0; t < contours.size(); t++) {
				//Ѱ����С����
				cv::RotatedRect currentrect = cv::minAreaRect(contours[t]);
				cv::Rect rect = currentrect.boundingRect();

				//�ж���û�и�����������У�ֱ�����������Ǽ�⸸����
				if(hierarchy[t][2]>0) continue;

				//�ж�������ͼ���д������֮һ��С�����֮�ĵ�����Ϊ�������
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
					rectcount++;//��¼������ҪԤ��ĸ���
					tmpsrc = frame(rect);
					std::vector<float> fv;

					try
					{
						//HOG����
						hog_deal(tmpsrc, fv);
						cv::Mat one_row = cv::Mat::zeros(cv::Size(fv.size(), 1), CV_32FC1);
						for (int i = 0; i < fv.size(); i++) {
							one_row.at<float>(0, i) = fv[i];
						}
						//����svm��Ԥ��
						float result = svm->predict(one_row);
						printf("result:%f\n", result);
						//���������0˵��Ԥ��ƥ�䵽�ˣ����Ͼ���
						if (result > 0)
						{
							predict++;//��¼ƥ�����
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