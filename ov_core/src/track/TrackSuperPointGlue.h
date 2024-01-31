
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
                           HistogramMethod histmethod, int gridx, int gridy, int minpxdist)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist)
        {
            config_path = "/home/pranav/catkin_ws/src/superpoint_superglue/config/config.yaml";
		        model_dir =  "/home/pranav/catkin_ws/src/superpoint_superglue/weights/";
            Configs configs(config_path, model_dir);
		        width = configs.superglue_config.image_width;
		        height = configs.superglue_config.image_height;
            SuperPoint_Eng = std::make_shared<SuperPoint>(configs.superpoint_config);
            PRINT_ALL("Building SuperPoint Inference Engine. This may take some time...\n");
            std::cout<<"Building SuperPoint Inference Engine. This may take some time..."<<std::endl;
            if(!SuperPoint_Eng->build())
            {
                PRINT_ALL("Failed to build SuperPoint engine!!\n");
                return;
            }
            SuperGlue_Eng = std::make_shared<SuperGlue>(configs.superglue_config);
            PRINT_ALL("Building SuperGlue Inference Engine.\n");
            std::cout<<"Building SuperGlue Inference Engine."<<std::endl;
            if (!SuperGlue_Eng->build())
            {
                PRINT_ALL("Failed to build SuperGlue engine!!\n");
                return;
            }

            PRINT_ALL("SuperPoint and SuperGlue Engine build success!!\n");
            std::cout<<"SuperPoint and SuperGlue Engine build success!!"<<std::endl;
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

  void feed_monocular(const CameraData &message, size_t msg_id);
  /**
   * @brief Process new stereo pair of images
   * @param message Contains our timestamp, images, and camera ids
   * @param msg_id_left first image index in message data vector
   * @param msg_id_right second image index in message data vector
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
   */

  void perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                                  Eigen::Matrix<double,259,Eigen::Dynamic> &desc0, std::vector<size_t> &ids0);

  void perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                               std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, Eigen::Matrix<double,259,Eigen::Dynamic> &desc0,
                                               Eigen::Matrix<double,259,Eigen::Dynamic> &desc1, size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0,
                                               std::vector<size_t> &ids1);

  /**
   * @brief Find matches between two keypoint+descriptor sets.
   * @param pts0 first vector of keypoints
   * @param pts1 second vector of keypoints
   * @param desc0 first matrix of confidence scores+keypoints+descriptors
   * @param desc1 second matrix of confidence scores+keypoints+decriptors
   * @param id0 id of the first camera
   * @param id1 id of the second camera
   * @param matches vector of matches that we have found
   *
   */

  // Timing variables
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;
  int threshold;
  int grid_x;
  int grid_y;

  // Minimum pixel distance to be "far away enough" to be a different extracted feature
  int min_px_dist;

  // Confidence Score + Keypoints + Descriptor matrices for SuperGlue matching.
  std::unordered_map<size_t, Eigen::Matrix<double,259,Eigen::Dynamic>> desc_last;
  // Path to config file of SuperPoint and SuperGlue
  std::string config_path;
  // Path to SuperPoint and SuperGlue Inference Engine weights
  std::string model_dir;
  // Width and height of image to be sent to SuperPoint engine
  int SuperPoint_img_width,SuperPoint_img_height;
  // SuperPoint Inference Engine
  std::shared_ptr<SuperPoint> SuperPoint_Eng;
  // SuperGlue Inference Engine
  std::shared_ptr<SuperGlue> SuperGlue_Eng;
  // Image Width
  int width;
  // Image Height
  int height;
    
};  

} // namespace ov_core

#endif /* OV_CORE_TRACK_SUPPNTGLU_H */
