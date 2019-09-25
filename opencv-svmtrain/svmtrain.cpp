#include <iostream>
#include <opencv2/opencv.hpp>

//ѵ�������ļ�
std::string trainfile = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/hog_train.xml";
//�����������Ŀ¼
std::string positive_dir = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/00";
//�����������Ŀ¼
std::string negative_dir = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/01";

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

	//���������������
	std::vector<cv::String> postiveimgs;
	cv::glob(positive_dir, postiveimgs);
	//��⸺����������
	std::vector<cv::String> negativeimgs;
	cv::glob(negative_dir, negativeimgs);
	//��������������
	int trainnum = postiveimgs.size() + negativeimgs.size();

	//����ѵ������,���ݵõ��������������Ͷ���Ŀ�ȼ�������ǵ�ѵ������
	//   	����Ҫ��һ���������ʶ��ÿһ��Ŀ�궼��Ӧһ��һά����������
	//      �������һ����nά�����n����ƾ��Ϲ�µģ��������оݣ�����ȷ���
	//      Ϊʲôopencv�Դ���hog�������3781ά�ģ���������ڳ���ȷʵ�Ƚ�ͷ�ۣ�
	//      �����˺ó���ʱ�䣬�������ż���������һ��opencv���HOGDescriptor����ṹ
	//      �Ĺ��캯��HOGDescriptor��Size winSize, Size blocksize, Size blockStride, 
	//      Size cellSize, ...(����Ĳ����������ò���)����ȥ��һ��opencvĬ�ϵĲ�������
	//      ���Կ� ����winSize��64, 128����blockSize��16, 16����blockStride��8, 8����
	//      cellSize��8, 8��������Ȼhog �ǽ�һ����������win����Ϊ�ܶ�Ŀ�block��
	//      ��ÿһ�������ֻ���Ϊ�ܶ��ϸ����Ԫcell(����Ԫ)��hog�����������ǰ���Щ����
	//      ��cell��Ӧ ��С�����������õ�һ����ά��������������ô������ڶ�Ӧ��һά
	//      ��������ά��n�͵��ڴ����еĿ��� x ���еİ�Ԫ��  x ÿһ����Ԫ��Ӧ��������������
	//		д����������еĿ��� x ���еİ�Ԫ��  x ÿһ����Ԫ��Ӧ������������, 
	//      ���뿴һ��n = 105x4x9 = 3780, �����������ڶ�Ӧ�������ˡ����˻�˵��
	//      Ϊʲôopencv���getDefaultPeopleDetector()�õ�����3781ά�أ�������Ϊ����һά
	//      ��һάƫ�ƣ����ܱ����ǰɣ���Ҳ�����ܾá���������һ�ν��ͣ���
	//		��������hog + svm������ˣ����յļ�ⷽ����������������б� ����wx + b = 0��
	//      �ղ������3780ά������ʵ����w��������һά��b���γ���opencvĬ�ϵ�3781ά������ӣ�
	//      ������Ϊtrain��test�����֣��� train�ڼ�������Ҫ��ȡһЩ��ѵ��������hog����
	//      ʹ��svmѵ�����յ�Ŀ����Ϊ�˵õ����Ǽ���w�Լ�b����test�ڼ���ȡ�����Ŀ���hog��
	//      ��x�����뷽���ǲ��Ǿ��ܽ����б����أ�
	cv::Mat trainData = cv::Mat::zeros(cv::Size(3780, trainnum), CV_32FC1);
	cv::Mat labels = cv::Mat::zeros(cv::Size(1, trainnum), CV_32SC1);

	try
	{
		//������������
		for (int i = 0; i < postiveimgs.size(); i++)
		{
			//����ͼƬ
			cv::Mat img = cv::imread(postiveimgs[i].c_str());
			printf("postiveimg:%s\n", postiveimgs[i].c_str());
			std::vector<float> fv;
			//����������
			hog_deal(img, fv);
			for (int j = 0; j < fv.size(); j++)
			{
				trainData.at<float>(i, j) = fv[j];
			}
			labels.at<float>(i, 0) = 1;
		}

		//����������
		for (int i = 0; i < negativeimgs.size(); i++)
		{
			//����ͼƬ
			cv::Mat img = cv::imread(negativeimgs[i].c_str());
			printf("negativeimg:%s\n", negativeimgs[i].c_str());
			std::vector<float> fv;
			//���������� 
			hog_deal(img, fv);
			for (int j = 0; j < fv.size(); j++)
			{
				trainData.at<float>(i + postiveimgs.size(), j) = fv[j];
			}
			labels.at<float>(i + postiveimgs.size(), 0) = -1;
		}

		//��ʼѵ������
		printf(" start SVM training \n");
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

		//����C��2.67Ĭ��ֵ������Խ�󣬼���Խ����
		svm->setC(2.67);
		//������SVC�ķ�����
		svm->setType(cv::ml::SVM::C_SVC);
		//�������Է��࣬����Ƚϼ򵥣��ٶȽϿ�
		svm->setKernel(cv::ml::SVM::LINEAR);
		//����Gamma,�����������������û���ã���Ҫ��������ķ���
		// svm->setGamma(5.383);

		//��ʼѵ��, ROW_SAMPLE���ǰ�����֯����

		svm->train(trainData, cv::ml::ROW_SAMPLE, labels);
		printf("end training \n");

		//����ѵ�������xml�ļ�
		svm->save(trainfile);
		printf("save training to %s\n", trainfile.c_str());
	}
	catch (const std::exception& ex)
	{
		printf(ex.what());
	}

	getchar();
	// cv::waitKey(0);

	return 0;
}
