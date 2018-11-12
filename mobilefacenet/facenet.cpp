#include <iostream>
#include "mobilefacenet.h"
#include "mtcnn.h"

int main(int argc, char** argv)
{
	MobileNetFeatureExtractor *pfe = new MobileNetFeatureExtractor("../models");
	std::vector<float> feature1;
    std::vector<float> feature2;
	std::vector<float> feature3;
	cv::Mat img1 = cv::imread("imgs/wang0.jpg");
	pfe->getFeature(img1, feature1);
	cv::Mat img2 = cv::imread("imgs/wang1.jpg");
	pfe->getFeature(img2, feature2);
	cv::Mat img3 = cv::imread("imgs/wang2.jpg");
	pfe->getFeature(img3, feature3);
	// double ss = calculSimilar(feature2, feature1);
	// std::cout << ss << std::endl;
	 // ss = calculSimilar(feature2, feature3);
	// std::cout << ss << std::endl;
	 // ss = calculSimilar(feature3, feature1);
	// std::cout << ss << std::endl;
	
	std::vector<float> feature4;
    std::vector<float> feature5;
	std::vector<float> feature6;
	cv::Mat img4 = cv::imread("imgs/le2.jpg");
	pfe->getFeature(img4, feature4);
	cv::Mat img5 = cv::imread("imgs/le3.jpg");
	pfe->getFeature(img5, feature5);
	cv::Mat img6 = cv::imread("imgs/le4.jpg");
	pfe->getFeature(img6, feature6);
    
	//test_detection();
    //testvalidation(argv[1], argv[2]);
	
	MTCNN mtcnn("../mmm");
	//cv::Mat image = cv::imread(argv[1]);
	
	cv::Mat image;
	cv::VideoCapture cam(argv[1]);
	while(1){ cam >> image;
	if(image.empty()) break;
	
	
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;
	mtcnn.detect(ncnn_img, finalBbox);
	
	const int num_box = finalBbox.size();
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);
	}
	
	std::cout << "============" << num_box << std::endl;
	
	// 身份认证
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++)
	{
		cv::Rect object = (*it);
		cv::Mat cut = image(object);
		rectangle(image, object, cv::Scalar(0, 0, 255), 2, 8, 0);
		

		//for(int i=0; i<cut.cols; i++)
		//{
		//	std::cout << float(img1.data[i]) << "   >>" << float(cut.data[i]) << std::endl;
		//}
		
		cv::imwrite("ss.jpg", cut);
		cv::Mat zz = cv::imread("ss.jpg");
		
		std::vector<float> feature;
		pfe->getFeature(zz, feature);
		
		double similarity = calculSimilar(feature, feature1);
		//std::cout << similarity << std::endl;
		similarity += calculSimilar(feature, feature2);
		//std::cout << similarity << std::endl;
		similarity += calculSimilar(feature, feature3);
		//std::cout << similarity << std::endl;
		
		if(similarity > 1.0)
		{
			std::string label = "wangwuyi";
			int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 4, &baseLine);
			
			rectangle(image, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::rectangle(image, cv::Rect(cv::Point(object.x, object.y- label_size.height),
                cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(image, label, cv::Point(object.x, object.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
		
		similarity = 0;
		similarity = calculSimilar(feature, feature4);
		similarity += calculSimilar(feature, feature5);
		similarity += calculSimilar(feature, feature6);
		if(similarity > 1.0)
		{
			std::string label = "huangle";
			int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 4, &baseLine);
			
			rectangle(image, (*it), cv::Scalar(0, 0, 255), 2, 8, 0);
			cv::rectangle(image, cv::Rect(cv::Point(object.x, object.y- label_size.height),
                cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(255, 255, 255), CV_FILLED);
            cv::putText(image, label, cv::Point(object.x, object.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
	}
	//cv::imwrite("result.jpg", image);
	cv::imshow("zz", image);
	cv::waitKey(30);
	
	}
	
	return 0;
}