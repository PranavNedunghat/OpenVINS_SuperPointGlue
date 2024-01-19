
#ifndef OV_CORE_TRACK_SUPPNTGLU_H
#define OV_CORE_TRACK_SUPPNTGLU_H

#include "TrackBase.h"
#include <Eigen/Dense>
#include "super_point.h"
#include "super_glue.h"
#include "utils.h"

namespace ov_core {

/**
 * @brief Descriptor-based visual tracking
 *
 * Here we use descriptor matching to track features from one frame to the next.
 * We track both temporally, and across stereo pairs to get stereo constraints.
 * This tracker uses the SuperPoint feature extractor Engine and SuperGlue feature matching Engine.
 */
class TrackSuperPointGlue : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param numfeats number of features we want want to track (i.e. track 200 points from frame to frame)
   * @param numaruco the max id of the arucotags, so we ensure that we start our non-auroc features above this value
   * @param stereo if we should do stereo feature tracking or binocular
   * @param histmethod what type of histogram pre-processing should be done (histogram eq?)
   * @param fast_threshold FAST detection threshold
   * @param gridx size of grid in the x-direction / u-direction
   * @param gridy size of grid in the y-direction / v-direction
   * @param minpxdist features need to be at least this number pixels away from each other
   * @param knnratio matching ratio needed (smaller value forces top two descriptors during match to be more different)
   */
  explicit TrackSuperPointGlue(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                           HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist, double knnratio)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist), knn_ratio(knnratio) 
        {
            config_path = "/home/pranav/ws_catkin/src/superpoint_superglue/config/config.yaml";
		        model_dir =  "/home/pranav/ws_catkin/src/superpoint_superglue/weights/";
            Configs configs(config_path, model_dir);
		        width = configs.superglue_config.image_width;
		        height = configs.superglue_config.image_height;
            SuperPoint_Eng0 = std::make_shared<SuperPoint>(configs.superpoint_config);
            SuperPoint_Eng1 = std::make_shared<SuperPoint>(configs.superpoint_config);
            PRINT_ALL("Building SuperPoint Inference Engine. This may take some time...\n");
            if(!SuperPoint_Eng0->build() || !SuperPoint_Eng1->build())
            {
                PRINT_ALL("Failed to build SuperPoint engine!!\n");
                return;
            }
            SuperGlue_Eng0 = std::make_shared<SuperGlue>(configs.superglue_config);
            SuperGlue_Eng1 = std::make_shared<SuperGlue>(configs.superglue_config);
            PRINT_ALL("Building SuperGlue Inference Engine.\n");
            if (!SuperGlue_Eng0->build() || !SuperGlue_Eng1->build())
            {
                PRINT_ALL("Failed to build SuperGlue engine!!\n");
                return;
            }

            PRINT_ALL("SuperPoint and SuperGlue Engine build success!!\n");
        }

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief Process a new monocular image
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id the camera index in message data vector
   */

  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief Detects new features in the current image
   * @param img0 image we will detect features on
   * @param mask0 mask which has what ROI we do not want features in
   * @param pts0 vector of extracted keypoints
   * @param desc0 vector of the extracted descriptors
   * @param ids0 vector of all new IDs
   *
   * Given a set of images, and their currently extracted features, this will try to add new features.
   * We return all extracted descriptors here since we DO NOT need to do stereo tracking left to right.
   * Our vector of IDs will be later overwritten when we match features temporally to the previous frame's features.
   * See robust_match() for the matching.
   */

  void perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, cv::Mat &desc0, cv::Mat &desc1,
                                size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  /**
   * @brief Find matches between two keypoint+descriptor sets.
   * @param pts0 first vector of keypoints
   * @param pts1 second vector of keypoints
   * @param desc0 first vector of descriptors
   * @param desc1 second vector of decriptors
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param matches vector of matches that we have found
   *
   * This will perform a "robust match" between the two sets of points (slow but has great results).
   * First we do a simple KNN match from 1to2 and 2to1, which is followed by a ratio check and symmetry check.
   * Original code is from the "RobustMatcher" in the opencv examples, and seems to give very good results in the matches.
   * https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
   */
  void robust_match(const std::vector<cv::KeyPoint> &pts0, const std::vector<cv::KeyPoint> &pts1, const cv::Mat &desc0,
                    const cv::Mat &desc1, size_t id0, size_t id1, std::vector<cv::DMatch> &matches);

  // Helper functions for the robust_match function
  // Original code is from the "RobustMatcher" in the opencv examples
  // https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
  void robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches);
  void robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                            std::vector<cv::DMatch> &good_matches);

  // Timing variables
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // Our orb extractor
  cv::Ptr<cv::ORB> orb0 = cv::ORB::create();
  cv::Ptr<cv::ORB> orb1 = cv::ORB::create();

  // Our descriptor matcher
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  // Parameters for our FAST grid detector
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // The ratio between two kNN matches, if that ratio is larger then this threshold
  // then the two features are too close, so should be considered ambiguous/bad match
  double knn_ratio;

  // Confidence Score + Keypoints + Descriptor matrices for SuperGlue matching.
  std::unordered_map<size_t, Eigen<double,259,Eigen::Dynamic>> desc_last;
  // Path to config file of SuperPoint and SuperGlue
  std::string config_path;
  // Path to SuperPoint and SuperGlue Inference Engine weights
  std::string model_dir;
  // Width and height of image to be sent to SuperPoint engine
  int SuperPoint_img_width,SuperPoint_img_height;
  // SuperPoint Inference Engine
  std::shared_ptr<SuperPoint> SuperPoint_Eng0,SuperPoint_Eng1;
  // SuperGlue Inference Engine
  std::shared_ptr<SuperGlue> SuperGlue_Eng0,SuperGlue_Eng1;
    
};  

} // namespace ov_core

#endif /* OV_CORE_TRACK_SUPPNTGLU_H */