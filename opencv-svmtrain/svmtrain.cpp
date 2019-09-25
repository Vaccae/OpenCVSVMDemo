#include <iostream>
#include <opencv2/opencv.hpp>

//训练生成文件
std::string trainfile = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/hog_train.xml";
//正向样本存放目录
std::string positive_dir = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/00";
//负向样本存放目录
std::string negative_dir = "D:/Business/DemoTEST/CPP/opencv-svm/svmresource/01";

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

	//检测正向样本个数
	std::vector<cv::String> postiveimgs;
	cv::glob(positive_dir, postiveimgs);
	//检测负向样本个数
	std::vector<cv::String> negativeimgs;
	cv::glob(negative_dir, negativeimgs);
	//定义总样本个数
	int trainnum = postiveimgs.size() + negativeimgs.size();

	//创建训练数据,根据得到的总样本个数和定义的宽度计算出我们的训练数据
	//   	首先要有一个整体的认识，每一个目标都对应一个一维特征向量，
	//      这个向量一共有n维，这个n不是凭空瞎猜的，是有理有据，打个比方，
	//      为什么opencv自带的hog检测子是3781维的？这个问题在初期确实比较头疼，
	//      纠结了好长的时间，不过别着急，先来看一下opencv里的HOGDescriptor这个结构
	//      的构造函数HOGDescriptor（Size winSize, Size blocksize, Size blockStride, 
	//      Size cellSize, ...(后面的参数在这里用不到)），去查一下opencv默认的参数我们
	//      可以看 到，winSize（64, 128），blockSize（16, 16），blockStride（8, 8），
	//      cellSize（8, 8），很显然hog 是将一个特征窗口win划分为很多的块block，
	//      在每一个块里又划分为很多的细胞单元cell(即胞元)，hog特征向量既是把这些所有
	//      的cell对应 的小特征串起来得到一个高维的特征向量，那么这个窗口对应的一维
	//      特征向量维数n就等于窗口中的块数 x 块中的胞元数  x 每一个胞元对应的特征向量数。
	//		写到这里，窗口中的块数 x 块中的胞元数  x 每一个胞元对应的特征向量数, 
	//      带入看一下n = 105x4x9 = 3780, 这就是这个窗口对应的特征了。有人会说，
	//      为什么opencv里的getDefaultPeopleDetector()得到的是3781维呢？这是因为另外一维
	//      是一维偏移，（很崩溃是吧，我也崩溃很久。。。，下一段解释）。
	//		我们利用hog + svm检测行人，最终的检测方法是最基本的线性判别函 数，wx + b = 0，
	//      刚才所求的3780维向量其实就是w，而加了一维的b就形成了opencv默认的3781维检测算子，
	//      而检测分为train和test两部分，在 train期间我们需要提取一些列训练样本的hog特征
	//      使用svm训练最终的目的是为了得到我们检测的w以及b，在test期间提取待检测目标的hog特
	//      征x，带入方程是不是就能进行判别了呢？
	cv::Mat trainData = cv::Mat::zeros(cv::Size(3780, trainnum), CV_32FC1);
	cv::Mat labels = cv::Mat::zeros(cv::Size(1, trainnum), CV_32SC1);

	try
	{
		//处理正向数据
		for (int i = 0; i < postiveimgs.size(); i++)
		{
			//加载图片
			cv::Mat img = cv::imread(postiveimgs[i].c_str());
			printf("postiveimg:%s\n", postiveimgs[i].c_str());
			std::vector<float> fv;
			//计算描述子
			hog_deal(img, fv);
			for (int j = 0; j < fv.size(); j++)
			{
				trainData.at<float>(i, j) = fv[j];
			}
			labels.at<float>(i, 0) = 1;
		}

		//处理负向数据
		for (int i = 0; i < negativeimgs.size(); i++)
		{
			//加载图片
			cv::Mat img = cv::imread(negativeimgs[i].c_str());
			printf("negativeimg:%s\n", negativeimgs[i].c_str());
			std::vector<float> fv;
			//计算描述子 
			hog_deal(img, fv);
			for (int j = 0; j < fv.size(); j++)
			{
				trainData.at<float>(i + postiveimgs.size(), j) = fv[j];
			}
			labels.at<float>(i + postiveimgs.size(), 0) = -1;
		}

		//开始训练数据
		printf(" start SVM training \n");
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

		//设置C的2.67默认值，设置越大，计算越复杂
		svm->setC(2.67);
		//设置用SVC的分类器
		svm->setType(cv::ml::SVM::C_SVC);
		//设置线性分类，计算比较简单，速度较快
		svm->setKernel(cv::ml::SVM::LINEAR);
		//设置Gamma,在上面用线性里这个没作用，主要用在另外的分类
		// svm->setGamma(5.383);

		//开始训练, ROW_SAMPLE就是按行组织样本

		svm->train(trainData, cv::ml::ROW_SAMPLE, labels);
		printf("end training \n");

		//保存训练结果进xml文件
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
