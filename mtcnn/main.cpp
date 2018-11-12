#include <opencv2/opencv.hpp>
#include "mtcnn.h"
using namespace cv;

cv::Mat drawDetection(const cv::Mat &img, std::vector<Bbox> &box)
{
    cv::Mat show = img.clone();
    const int num_box = box.size();
    std::vector<cv::Rect> bbox;
    bbox.resize(num_box);
    for (int i = 0; i < num_box; i++) {
        bbox[i] = cv::Rect(box[i].x1, box[i].y1, box[i].x2 - box[i].x1 + 1, box[i].y2 - box[i].y1 + 1);

        for (int j = 0; j < 5; j = j + 1)
        {
            cv::circle(show, cvPoint(box[i].ppoint[j], box[i].ppoint[j + 5]), 2, CV_RGB(0, 255, 0), CV_FILLED);
        }
    }
    for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
        rectangle(show, (*it), Scalar(0, 0, 255), 2, 8, 0);
    }
    return show;
}


int main(int argc, char** argv)
{
    cv::namedWindow("img", CV_WINDOW_NORMAL);
	
    string model_path = ".";
    MTCNN mtcnn;
    mtcnn.init(model_path);
    mtcnn.SetMinFace(40);
	
    cv::Mat frame;
	frame = imread("3.jpg");
	
     
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
	std::vector<Bbox> finalBbox;

	mtcnn.detectMaxFace(ncnn_img, finalBbox);
	//mtcnn.detect(ncnn_img, finalBbox);

	cv::Mat show = drawDetection(frame, finalBbox);
	cv::imshow("img", show);
	cv::waitKey(0);
	
    return 0;
}