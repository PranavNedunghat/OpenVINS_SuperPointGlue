#include "ros/ros.h"
#include "ros/package.h"
#include "super_point.h"
#include "super_glue.h"
#include "utils.h"
#include "sensor_msgs/CompressedImage.h"
#include <memory>
#include <chrono>

class SuperPointGlue
{
	public:
	ros::NodeHandle node;
	ros::Subscriber sub;
	std::shared_ptr<sensor_msgs::CompressedImage> image1;
	std::shared_ptr<sensor_msgs::CompressedImage> image2;
	std::shared_ptr<SuperPoint> superpoint;
	std::shared_ptr<SuperGlue> superglue;	
	int width,height;
	SuperPointGlue()
	{
		std::string config_path = "/home/race8/ws_catkin/src/superpoint_superglue/config/config.yaml";
		std::string model_dir =  "/home/race8/ws_catkin/src/superpoint_superglue/weights/";
		Configs configs(config_path, model_dir);
		width = configs.superglue_config.image_width;
		height = configs.superglue_config.image_height;
		ROS_INFO("Building Inference Engine\n");
		superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
		if(!superpoint->build())
		{
			ROS_INFO("Failed to build SuperPoint engine!!\n");
			return;
		}
		superglue = std::make_shared<SuperGlue>(configs.superglue_config);
		if (!superglue->build())
		{
			ROS_INFO("Failed to build SuperGlue engine!!\n");
			return;
		}

		ROS_INFO("SuperPoint and SuperGlue engine build success!!\n");
		node = ros::NodeHandle("SuperPointGlue_node");
		sub = node.subscribe("/race12/cam1/color/image_raw/compressed",1,&SuperPointGlue::matches_callback,this);
	}
	void matches_callback(const sensor_msgs::CompressedImage::ConstPtr& msg)
	{
		image1 = std::make_shared<sensor_msgs::CompressedImage>(*msg);
		if (image1 && image2)
		{	
			Eigen::Matrix<double,259,Eigen::Dynamic> feature_points1;
			Eigen::Matrix<double,259,Eigen::Dynamic> feature_points2;
			cv::Mat Im1(cv::imdecode(image1->data, cv::IMREAD_GRAYSCALE));
			cv::Mat Im2(cv::imdecode(image2->data, cv::IMREAD_GRAYSCALE));
			if(Im1.empty() || Im2.empty())
			{
				ROS_INFO("Image1 or Image2 empty!!\n");
				return;
			}
			//Im1 = cv::cvtColor(Im1, cv::RGB2GRAY);
			cv::resize(Im1, Im1, cv::Size(width, height));
			if(!superpoint->infer(Im1,feature_points1))
			{
				ROS_INFO("Failed when extracting features from first Image\n");
				return;
			}
			std::vector<cv::DMatch> _matches;
			//Im2 = cv::cvtColor(Im2,cv::RGB2GRAY)
			cv::resize(Im2,Im2,cv::Size(width,height));
			auto start = std::chrono::high_resolution_clock::now();
			if(!superpoint->infer(Im2,feature_points2))
			{
				ROS_INFO("Failed when extracting features from second Image\n");
				return;
			}
			if(feature_points1.cols()<=0 || feature_points2.cols()<=0)
			{
				ROS_INFO("No features in frame!! Skipping this frame!\n");
				return;
			}
			superglue->matching_points(feature_points1, feature_points2, _matches);
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
			cv::Mat match_image;
			std::vector<cv::KeyPoint> keypoints1, keypoints2;
			for(size_t i = 0; i < feature_points1.cols(); i++)
			{
				double score = feature_points1(0,i);
				double x = feature_points1(1,i);
				double y = feature_points1(2,i);
				keypoints1.emplace_back(x,y,8,-1,score);
			}
			for(size_t i = 0; i < feature_points2.cols(); i++)
			{
				double score = feature_points2(0,i);
				double x = feature_points2(1,i);
				double y = feature_points2(2,i);
				keypoints2.emplace_back(x,y, 8, -1, score);
			}
			VisualizeMatching(Im1,keypoints1, Im2, keypoints2, _matches, match_image,duration.count());
			cv::imshow("Feature Matching",match_image);
			cv::waitKey(1);


		}
		image2 = std::make_shared<sensor_msgs::CompressedImage>(*image1);
	}

	


};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "superpointglue_node");
	SuperPointGlue superpointglue_node;
	ros::spin();
}


