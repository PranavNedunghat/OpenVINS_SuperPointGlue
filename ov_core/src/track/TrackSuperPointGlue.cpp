
#include "TrackSuperPointGlue.h"

#include <opencv2/features2d.hpp>
#include <Eigen/Dense>
#include "Grider_FAST.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "super_point.h"
#include "super_glue.h"
#include "utils.h"

using namespace ov_core;

void TrackSuperPointGlue::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() || message.sensor_ids.size() != message.images.size() || message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

  // If we are doing binocular tracking, then we should parallize our tracking
  size_t num_images = message.images.size();
 if (num_images == 1) {
    feed_monocular(message, 0);
  } 
 else if (num_images == 2 && use_stereo) {
    feed_stereo(message, 0, 1);
  }
 else {
    PRINT_ERROR(RED "[ERROR]: invalid number of images passed %zu, SuperPointGlue Tracker only supports monocular or stereo images at this time\n", num_images);
    std::exit(EXIT_FAILURE);
  }
}

void TrackSuperPointGlue::feed_monocular(const CameraData &message, size_t msg_id) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id = message.sensor_ids.at(msg_id);
  std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

  // Histogram equalize
  cv::Mat img, mask;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id), img);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id), img);
  } else {
    img = message.images.at(msg_id);
  }
  mask = message.masks.at(msg_id);

  // If we are the first frame (or have lost tracking), initialize our descriptors
  if (pts_last.find(cam_id) == pts_last.end() || pts_last[cam_id].empty()) {
    std::vector<cv::KeyPoint> good_left;
    std::vector<size_t> good_ids_left;
    Eigen::Matrix<double,259,Eigen::Dynamic> good_desc_left;
    perform_detection_monocular(img, mask, good_left, good_desc_left, good_ids_left);
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    desc_last[cam_id] = good_desc_left;
    return;
  }

  // Our new keypoints and descriptor for the new image
  std::vector<cv::KeyPoint> pts_new;
  Eigen::Matrix<double,259,Eigen::Dynamic> desc_new;
  std::vector<size_t> ids_new;

  // First, extract new descriptors for this new image
  perform_detection_monocular(img, mask, pts_new, desc_new, ids_new);
  rT2 = boost::posix_time::microsec_clock::local_time();

  // Our matches temporally
  std::vector<cv::DMatch> matches_ll;

  // Lets match temporally
  // Do a check before using the SuperGlue Engine to ensure there are features. Else, SuperGlue will give a segmentation fault error. 
  // WARNING: This would mean that we will have to re-initialize our descriptors if no features are found in the current set of images. 
  if(desc_new.cols()>0)
  {

  SuperGlue_Eng->matching_points(desc_last[cam_id],desc_new,matches_ll);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();
  std::cout<<"GOT "<<matches_ll.size()<<" MATCHES IN CURRENT FRAME\n"<<std::endl;
  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left;
  std::vector<size_t> good_ids_left;
  Eigen::Matrix<double,259,Eigen::Dynamic> good_desc_left;

  // Count how many we have tracked from the last time
  int num_tracklast = 0;

  // Resize the Eigen::Matrix vectors so that we can populate it with the score_points+descriptors and add it to our database
  int num_cols_new_desc = pts_new.size();
  good_desc_left.resize(259,num_cols_new_desc);

  // Loop through all current left to right points
  // We want to see if any of theses have matches to the previous frame
  // If we have a match new->old then we want to use that ID instead of the new one
  for (size_t i = 0; i < pts_new.size(); i++) {

    // Loop through all left matches, and find the old "train" id
    int idll = -1;
    for (size_t j = 0; j < matches_ll.size(); j++) {
      if (matches_ll[j].trainIdx == (int)i) {
        idll = matches_ll[j].queryIdx;
      }
    }

    // Then lets replace the current ID with the old ID if found
    // Else just append the current feature and its unique ID
    good_left.push_back(pts_new[i]);
    good_desc_left.col(i) = desc_new.col(i);
    if (idll != -1) {
      good_ids_left.push_back(ids_last[cam_id][idll]);
      num_tracklast++;
    } else {
      good_ids_left.push_back(ids_new[i]);
    }
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    cv::Point2f npt_l = camera_calib.at(cam_id)->undistort_cv(good_left.at(i).pt);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x, npt_l.y);
  }

  // Debug info
  // PRINT_DEBUG("LtoL = %d | good = %d | fromlast = %d\n",(int)matches_ll.size(),(int)good_left.size(),num_tracklast);

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id] = img;
    img_mask_last[cam_id] = mask;
    pts_last[cam_id] = good_left;
    ids_last[cam_id] = good_ids_left;
    desc_last[cam_id] = good_desc_left;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Our timing information
  PRINT_ALL("[TIME-DESC]: %.4f seconds for detection\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for matching\n", (rT3 - rT2).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for merging\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-DESC]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackSuperPointGlue::feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right) {

  // Start timing
  rT1 = boost::posix_time::microsec_clock::local_time();

  // Lock this data feed for this camera
  size_t cam_id_left = message.sensor_ids.at(msg_id_left);
  size_t cam_id_right = message.sensor_ids.at(msg_id_right);
  std::lock_guard<std::mutex> lck1(mtx_feeds.at(cam_id_left));
  std::lock_guard<std::mutex> lck2(mtx_feeds.at(cam_id_right));

  // Histogram equalize images
  cv::Mat img_left, img_right, mask_left, mask_right;
  if (histogram_method == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(message.images.at(msg_id_left), img_left);
    cv::equalizeHist(message.images.at(msg_id_right), img_right);
  } else if (histogram_method == HistogramMethod::CLAHE) {
    double eq_clip_limit = 10.0;
    cv::Size eq_win_size = cv::Size(8, 8);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
    clahe->apply(message.images.at(msg_id_left), img_left);
    clahe->apply(message.images.at(msg_id_right), img_right);
  } else {
    img_left = message.images.at(msg_id_left);
    img_right = message.images.at(msg_id_right);
  }
  mask_left = message.masks.at(msg_id_left);
  mask_right = message.masks.at(msg_id_right);

  // If we are the first frame (or have lost tracking), initialize our descriptors
  if (pts_last[cam_id_left].empty() || pts_last[cam_id_right].empty()) {
    std::vector<cv::KeyPoint> good_left, good_right;
    std::vector<size_t> good_ids_left, good_ids_right;
    Eigen::Matrix<double,259,Eigen::Dynamic> good_desc_left, good_desc_right;
    perform_detection_stereo(img_left, img_right, mask_left, mask_right, good_left, good_right, good_desc_left, good_desc_right,
                             cam_id_left, cam_id_right, good_ids_left, good_ids_right);
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    desc_last[cam_id_left] = good_desc_left;
    desc_last[cam_id_right] = good_desc_right;
    return;
  }

  // Our new keypoints and descriptor for the new image
  std::vector<cv::KeyPoint> pts_left_new, pts_right_new;
  Eigen::Matrix<double,259,Eigen::Dynamic> desc_left_new, desc_right_new;
  std::vector<size_t> ids_left_new, ids_right_new;

  // First, extract new descriptors for this new image
  perform_detection_stereo(img_left, img_right, mask_left, mask_right, pts_left_new, pts_right_new, desc_left_new, desc_right_new,
                           cam_id_left, cam_id_right, ids_left_new, ids_right_new);
  rT2 = boost::posix_time::microsec_clock::local_time();
  std::vector<cv::DMatch> matches_ll, matches_rr;
  // Do a check before using the SuperGlue Engine to ensure there are features. Else, SuperGlue will give a segmentation fault error. 
  // WARNING: This would mean that we will have to re-initialize our descriptors if no features are found in the current set of images. 
  if(desc_left_new.cols()>0 && desc_right_new.cols()>0)
  {

  SuperGlue_Eng->matching_points(desc_last[cam_id_left],desc_left_new,matches_ll);
  SuperGlue_Eng->matching_points(desc_last[cam_id_right],desc_right_new,matches_rr);
  }
  rT3 = boost::posix_time::microsec_clock::local_time();

  // Get our "good tracks"
  std::vector<cv::KeyPoint> good_left, good_right;
  std::vector<size_t> good_ids_left, good_ids_right;
  Eigen::Matrix<double,259,Eigen::Dynamic> good_desc_left, good_desc_right;

  // Points must be of equal size
  assert(pts_last[cam_id_left].size() == pts_last[cam_id_right].size());
  assert(pts_left_new.size() == pts_right_new.size());

  // Count how many we have tracked from the last time
  int num_tracklast = 0;

  // Resize the Eigen::Matrix vectors so that we can populate it with the score_points+descriptors and add it to our database
  int num_cols_new_desc = pts_left_new.size();
  good_desc_left.resize(259,num_cols_new_desc);
  good_desc_right.resize(259,num_cols_new_desc);

  // Loop through all current left to right points
  // We want to see if any of theses have matches to the previous frame
  // If we have a match new->old then we want to use that ID instead of the new one
  for (size_t i = 0; i < pts_left_new.size(); i++) {

    // Loop through all left matches, and find the old "train" id
    int idll = -1;
    for (size_t j = 0; j < matches_ll.size(); j++) {
      if (matches_ll[j].trainIdx == (int)i) {
        idll = matches_ll[j].queryIdx;
      }
    }

    // Loop through all left matches, and find the old "train" id
    int idrr = -1;
    for (size_t j = 0; j < matches_rr.size(); j++) {
      if (matches_rr[j].trainIdx == (int)i) {
        idrr = matches_rr[j].queryIdx;
      }
    }

    // If we found a good stereo track from left to left, and right to right
    // Then lets replace the current ID with the old ID
    // We also check that we are linked to the same past ID value
    if (idll != -1 && idrr != -1 && ids_last[cam_id_left][idll] == ids_last[cam_id_right][idrr]) {
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[i]);
      good_desc_left.col(i) = desc_left_new.col(i);
      good_desc_right.col(i) = desc_right_new.col(i);
      good_ids_left.push_back(ids_last[cam_id_left][idll]);
      good_ids_right.push_back(ids_last[cam_id_right][idrr]);
      num_tracklast++;
    } else {
      // Else just append the current feature and its unique ID
      good_left.push_back(pts_left_new[i]);
      good_right.push_back(pts_right_new[i]);
      good_desc_left.col(i) = desc_left_new.col(i);
      good_desc_right.col(i) = desc_right_new.col(i);
      good_ids_left.push_back(ids_left_new[i]);
      good_ids_right.push_back(ids_left_new[i]);
    }
  }
  rT4 = boost::posix_time::microsec_clock::local_time();

  //===================================================================================
  //===================================================================================

  // Update our feature database, with theses new observations
  for (size_t i = 0; i < good_left.size(); i++) {
    // Assert that our IDs are the same
    assert(good_ids_left.at(i) == good_ids_right.at(i));
    // Try to undistort the point
    cv::Point2f npt_l = camera_calib.at(cam_id_left)->undistort_cv(good_left.at(i).pt);
    cv::Point2f npt_r = camera_calib.at(cam_id_right)->undistort_cv(good_right.at(i).pt);
    // Append to the database
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_left, good_left.at(i).pt.x, good_left.at(i).pt.y, npt_l.x,
                             npt_l.y);
    database->update_feature(good_ids_left.at(i), message.timestamp, cam_id_right, good_right.at(i).pt.x, good_right.at(i).pt.y, npt_r.x,
                             npt_r.y);
  }

  // Debug info
  // PRINT_DEBUG("LtoL = %d | RtoR = %d | LtoR = %d | good = %d | fromlast = %d\n", (int)matches_ll.size(),
  //       (int)matches_rr.size(),(int)ids_left_new.size(),(int)good_left.size(),num_tracklast);

  // Move forward in time
  {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    img_last[cam_id_left] = img_left;
    img_last[cam_id_right] = img_right;
    img_mask_last[cam_id_left] = mask_left;
    img_mask_last[cam_id_right] = mask_right;
    pts_last[cam_id_left] = good_left;
    pts_last[cam_id_right] = good_right;
    ids_last[cam_id_left] = good_ids_left;
    ids_last[cam_id_right] = good_ids_right;
    desc_last[cam_id_left] = good_desc_left;
    desc_last[cam_id_right] = good_desc_right;
  }
  rT5 = boost::posix_time::microsec_clock::local_time();

  // Our timing information
  PRINT_ALL("[TIME-DESC]: %.4f seconds for detection\n", (rT2 - rT1).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for matching\n", (rT3 - rT2).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for merging\n", (rT4 - rT3).total_microseconds() * 1e-6);
  PRINT_ALL("[TIME-DESC]: %.4f seconds for feature DB update (%d features)\n", (rT5 - rT4).total_microseconds() * 1e-6,
            (int)good_left.size());
  PRINT_ALL("[TIME-DESC]: %.4f seconds for total\n", (rT5 - rT1).total_microseconds() * 1e-6);
}

void TrackSuperPointGlue::perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0,
                                                  Eigen::Matrix<double,259,Eigen::Dynamic> &desc0, std::vector<size_t> &ids0) {

  // Assert that we need features
  assert(pts0.empty());

  Eigen::Matrix<double,259,Eigen::Dynamic> pts_desc_0; 
  cv::Mat Im0;

  //Resize the images to fit with the requirements of the Inference Engines
  //NOTE: This is redundant if the Engines expect the image dimensions to be the resolution of the camera. 
  //TODO: If the resolution and image dimensions expected by the Engines are different, we should scale the keypoints back to the camera resolution scale.
  cv::resize(img0, Im0, cv::Size(width, height));

  // Extract our features (using the SuperPoint Engine, instead of ORB as used by OpenVINS), and their descriptors as an Eigen Matrix.
  // Note: SuperPoint extracts keypoints and descriptors and stores them together along with a confidence score in an Eigen Matrix.
  // The dimensions of the Matrix are 259 x num_keypoints_detected. The first row contains the confidence score, followed by the x, y pixel
  // coordinates in the second and third rows, respectively. The final 256 rows store the descriptors for each point to be used by SuperGlue.
  std::vector<cv::KeyPoint> pts0_ext;

  if(!SuperPoint_Eng->infer(Im0,pts_desc_0))
  {
    PRINT_WARNING("Failed to extract features from Camera 0!\n");
		return;
  }

  // Extract KeyPoints and the confidence score from the Eigen Matrix and store them in a KeyPoints vector to be stored in the OpenVINS database of features.
 
  for(size_t i = 0; i < pts_desc_0.cols(); i++)
			{
				double score = pts_desc_0(0,i);
				double x = pts_desc_0(1,i);
				double y = pts_desc_0(2,i);
				pts0_ext.emplace_back(x,y,8,-1,score);
			}

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size((int)((float)img0.cols / (float)min_px_dist), (int)((float)img0.rows / (float)min_px_dist));
  cv::Mat grid_2d = cv::Mat::zeros(size, CV_8UC1);


  // Store the indexes of the matched points to be used to create returning Eigen::Matrices

  std::vector<size_t> pts0Vec;

  // For all good matches, lets append to our returned vectors
  // NOTE: if we multi-thread this atomic can cause some randomness due to multiple thread detecting features
  // NOTE: this is due to the fact that we select update features based on feat id
  // NOTE: thus the order will matter since we try to select oldest (smallest id) to update with
  // NOTE: not sure how to remove... maybe a better way?
  for (size_t i = 0; i < pts0_ext.size(); i++) {
    // Get current left keypoint, check that it is in bounds
    cv::KeyPoint kpt = pts0_ext.at(i);
    int x = (int)kpt.pt.x;
    int y = (int)kpt.pt.y;
    int x_grid = (int)(kpt.pt.x / (float)min_px_dist);
    int y_grid = (int)(kpt.pt.y / (float)min_px_dist);
    if (x_grid < 0 || x_grid >= size.width || y_grid < 0 || y_grid >= size.height || x < 0 || x >= img0.cols || y < 0 || y >= img0.rows) {
      continue;
    }
    // Check if this keypoint is near another point
    if (grid_2d.at<uint8_t>(y_grid, x_grid) > 127)
      continue;
    // Else we are good, append our keypoints and descriptors
    pts0.push_back(pts0_ext.at(i));
    pts0Vec.push_back(i);
    // Set our IDs to be unique IDs here, will later replace with corrected ones, after temporal matching
    size_t temp = ++currid;
    ids0.push_back(temp);
    grid_2d.at<uint8_t>(y_grid, x_grid) = 255;
  }

  // Store the selected indices' confidence scores+keypoints+descriptors In the Eigen Matrix. This needs to be done seperately since Eigen requires the 
  // dimenstions of the matrix to be known before we can store data.
  int num_cols_new = pts0Vec.size();
  desc0.resize(259,num_cols_new);
  for(size_t i = 0; i < pts0Vec.size(); i++)
  {
    desc0.col(i) = pts_desc_0.col(pts0Vec[i]);
  }
}

void TrackSuperPointGlue::perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                               std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, Eigen::Matrix<double,259,Eigen::Dynamic> &desc0,
                                               Eigen::Matrix<double,259,Eigen::Dynamic> &desc1, size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0,
                                               std::vector<size_t> &ids1) {

  // Assert that we need features
  assert(pts0.empty());
  assert(pts1.empty());

  Eigen::Matrix<double,259,Eigen::Dynamic> pts_desc_0, pts_desc_1; 

  // Assumption: The original OpenVINS implementation preprocesses the images using GriderFAST algorithm. We will be skipping this, and SuperPoint should
  // still work properly.

  // Question: What is the use of these masks that is implemented in the Descriptor Tracker as seen in TrackDescriptor.cpp in this same function?

  //Resize the images to fit with the requirements of the Inference Engines
  //NOTE: This is redundant if the Engines expect the image dimensions to be the resolution of the camera. 
  //TODO: If the resolution and image dimensions expected by the Engines are different, we should scale the keypoints back to the camera resolution scale.
  cv::Mat Im0, Im1, match_image;
  cv::resize(img0, Im0, cv::Size(width, height));
  cv::resize(img1, Im1, cv::Size(width, height));
  std::vector<cv::KeyPoint> pts0_ext, pts1_ext;

  // Extract our features (using the SuperPoint Engine, instead of ORB as used by OpenVINS), and their descriptors as an Eigen Matrix.
  // Note: SuperPoint extracts keypoints and descriptors and stores them together along with a confidence score in an Eigen Matrix.
  // The dimensions of the Matrix are 259 x num_keypoints_detected. The first row contains the confidence score, followed by the x, y pixel
  // coordinates in the second and third rows, respectively. The final 256 rows store the descriptors for each point to be used by SuperGlue.

  if(!SuperPoint_Eng->infer(Im0,pts_desc_0))
  {
    PRINT_WARNING("Failed to extract features from Camera 0!\n");
		return;
  }
  if(!SuperPoint_Eng->infer(Im1,pts_desc_1))
  {
    PRINT_WARNING("Failed to extract features from Camera 1!\n");
		return;
  }

  // Do matching from the left to the right image
  std::vector<cv::DMatch> matches;
  // Add a check to make sure that the Eigen Matrices are not empty. Else SuperGlue will give a segmentation fault.
  if(pts_desc_0.cols()>0 || pts_desc_1.cols()>0)
			{
				
  // Here, the SuperGlue Engine takes over from the robust_match function implemented for OpenVINS. The matching_points function does the same 
  // set of operations as robust match i.e. match points followed by RANSAC which filters the points and appends to the matches vector.
  // Assumption: SuperPoint outputs x,y coordinates in the pixel coordinate frame with top-left corner as the origin. This should be compatible with OpenVINS Tracking.

        SuperGlue_Eng->matching_points(pts_desc_0,pts_desc_1,matches);
      }

  //Question: Why do we need to match features across the left and right cameras?

  // Extract KeyPoints and the confidence score from the Eigen Matrices and store them in a KeyPoints vector to be stored in the OpenVINS database of features.
  // Note: Could be a bottleneck here. Perhaps use a parallel_for loop?
  for(size_t i = 0; i < pts_desc_0.cols(); i++)
			{
				double score = pts_desc_0(0,i);
				double x = pts_desc_0(1,i);
				double y = pts_desc_0(2,i);
				pts0_ext.emplace_back(x,y,8,-1,score);
			}
  for(size_t i = 0; i < pts_desc_1.cols(); i++)
			{
				double score = pts_desc_1(0,i);
				double x = pts_desc_1(1,i);
				double y = pts_desc_1(2,i);
				pts1_ext.emplace_back(x,y, 8, -1, score);
			}
  // Visualize the SuperPoint and SuperGlue engine outputs
  VisualizeMatching(Im0,pts0_ext, Im1, pts1_ext, matches, match_image);
  cv::imshow("Feature Matching",match_image);
	cv::waitKey(1);

  // Create a 2D occupancy grid for this current image
  // Note that we scale this down, so that each grid point is equal to a set of pixels
  // This means that we will reject points that less then grid_px_size points away then existing features
  cv::Size size0((int)((float)img0.cols / (float)min_px_dist), (int)((float)img0.rows / (float)min_px_dist));
  cv::Mat grid_2d_0 = cv::Mat::zeros(size0, CV_8UC1);
  cv::Size size1((int)((float)img1.cols / (float)min_px_dist), (int)((float)img1.rows / (float)min_px_dist));
  cv::Mat grid_2d_1 = cv::Mat::zeros(size1, CV_8UC1);

  // Store the indexes of the matched points to be used to create returning Eigen::Matrices

  std::vector<size_t> pts0Vec, pts1Vec;

  // For all good matches, lets append to our returned vectors
  for (size_t i = 0; i < matches.size(); i++) {

    // Get our ids
    int index_pt0 = matches.at(i).queryIdx;
    int index_pt1 = matches.at(i).trainIdx;

    // Get current left/right keypoint, check that it is in bounds
    //Question: Is this really necessary in the case of SuperPoint and SuperGlue?
    cv::KeyPoint kpt0 = pts0_ext.at(index_pt0);
    cv::KeyPoint kpt1 = pts1_ext.at(index_pt1);
    int x0 = (int)kpt0.pt.x;
    int y0 = (int)kpt0.pt.y;
    int x0_grid = (int)(kpt0.pt.x / (float)min_px_dist);
    int y0_grid = (int)(kpt0.pt.y / (float)min_px_dist);
    if (x0_grid < 0 || x0_grid >= size0.width || y0_grid < 0 || y0_grid >= size0.height || x0 < 0 || x0 >= img0.cols || y0 < 0 ||
        y0 >= img0.rows) {
      continue;
    }
    int x1 = (int)kpt1.pt.x;
    int y1 = (int)kpt1.pt.y;
    int x1_grid = (int)(kpt1.pt.x / (float)min_px_dist);
    int y1_grid = (int)(kpt1.pt.y / (float)min_px_dist);
    if (x1_grid < 0 || x1_grid >= size1.width || y1_grid < 0 || y1_grid >= size1.height || x1 < 0 || x1 >= img0.cols || y1 < 0 ||
        y1 >= img0.rows) {
      continue;
    }

    // Check if this keypoint is near another point
    if (grid_2d_0.at<uint8_t>(y0_grid, x0_grid) > 127 || grid_2d_1.at<uint8_t>(y1_grid, x1_grid) > 127)
      continue;

    // Append our keypoints and their column indices to be used to store the Eigen::Matrix scores+pixel_coords+desciptors

    pts0Vec.push_back(index_pt0);
    pts1Vec.push_back(index_pt1);
    pts0.push_back(pts0_ext[index_pt0]);
    pts1.push_back(pts1_ext[index_pt1]);

    // Set our IDs to be unique IDs here, will later replace with corrected ones, after temporal matching
    size_t temp = ++currid;
    ids0.push_back(temp);
    ids1.push_back(temp);
  }
  // Here we take the corresponding columns vectors from the Eigen::Matrix returned by SuperGlue and store the filtered scores+points+descriptors from
  // the matches in another Eigen::Matrix which is will be stored in the desc_last database to perform temporal matching.
  // This needs to be done seperately since Eigen requires the dimenstions of the matrix to be known before we can store data.
  int num_cols_new = pts0Vec.size();
  desc0.resize(259,num_cols_new);
  desc1.resize(259,num_cols_new);
  for(size_t i = 0; i < pts0Vec.size(); i++)
  {
    desc0.col(i) = pts_desc_0.col(pts0Vec[i]);
    desc1.col(i) = pts_desc_1.col(pts1Vec[i]);
  }
}
