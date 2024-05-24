#include <active_perception/perception_utils.h>

#include <pcl/filters/voxel_grid.h>

using namespace std;

namespace fast_planner {
PerceptionUtils::PerceptionUtils(ros::NodeHandle& nh) {
  nh.param("perception_utils/top_angle", top_angle_, -1.0);
  nh.param("perception_utils/left_angle", left_angle_, -1.0);
  nh.param("perception_utils/right_angle", right_angle_, -1.0);
  nh.param("perception_utils/max_dist", max_dist_, -1.0);
  nh.param("perception_utils/vis_dist", vis_dist_, -1.0);

  nh.param("map_ros/fx", fx, -1.0);
  nh.param("map_ros/fy", fy, -1.0);
  nh.param("map_ros/cx", cx, -1.0);
  nh.param("map_ros/cy", cy, -1.0);
  cameraIntrinsics << fx, 0.0, cx,
                        0.0, fy, cy,
                        0.0, 0.0, 1.0;
  imageSize << 2*cx,2*cy;

  // FOV vertices in body frame, for FOV visualization
  double hor = vis_dist_ * tan(left_angle_);
  double vert = vis_dist_ * tan(top_angle_);

  // by zj
  // Vector3d origin(0, 0, 0);
  // Vector3d left_up(-0.2, 0.2, -0.5);
  // Vector3d left_down(-0.2, -0.2, -0.5);
  // Vector3d right_up(0.2, 0.2, -0.5);
  // Vector3d right_down(0.2, -0.2, -0.5);

  Vector3d origin(0, 0, 0);
  Vector3d left_up(0.3, 0.3, 0.6);
  Vector3d left_down(0.3, -0.3, 0.6);
  Vector3d right_up(-0.3, 0.3, 0.6);
  Vector3d right_down(-0.3, -0.3, 0.6);


  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(left_up);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(left_down);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(right_up);
  cam_vertices1_.push_back(origin);
  cam_vertices2_.push_back(right_down);

  cam_vertices1_.push_back(left_up);
  cam_vertices2_.push_back(right_up);
  cam_vertices1_.push_back(right_up);
  cam_vertices2_.push_back(right_down);
  cam_vertices1_.push_back(right_down);
  cam_vertices2_.push_back(left_down);
  cam_vertices1_.push_back(left_down);
  cam_vertices2_.push_back(left_up);
}

void PerceptionUtils::setPose(const Vector3d& pos, const double& pitch, const double& yaw) {
  pos_ = pos;
  pitch_ = pitch;
  yaw_ = yaw;

}

void PerceptionUtils::getFOV(vector<Vector3d>& list1, vector<Vector3d>& list2) {
  list1.clear();
  list2.clear();

  // Get info for visualizing FOV at (pos, yaw)
  Eigen::Matrix3d Rwb;
  Eigen::Matrix4d cameraPose = get_pose(yaw_,pitch_,0,pos_);
  Rwb = cameraPose.block<3, 3>(0, 0);
  for (int i = 0; i < cam_vertices1_.size(); ++i) {
    auto p1 = Rwb * cam_vertices1_[i] + pos_;
    auto p2 = Rwb * cam_vertices2_[i] + pos_;
    list1.push_back(p1);
    list2.push_back(p2);
  }
}

bool PerceptionUtils::insideFOV(const Vector3d& point) {
  Eigen::Vector3d dir = point - pos_;
  if (dir.norm() > max_dist_) return false;
  
  // Eigen::Vector3d pos(1.48333,  1.55943, -2.03677);
  // double yaw = 0.527938;
  // cout << pos << endl;
  // Eigen::Matrix4d cam_pose = get_pose(yaw+M_PI/2.0*2,0,0,pos);
  // cout << cam_pose << endl;
  // Eigen::Vector3d point1(2.4,1.55943,-0.449057);
  // bool isInView = isPointInCameraView(point1,cam_pose,cameraIntrinsics,imageSize);
  // cout << "isInView = " << isInView << endl;

  
  Eigen::Matrix4d cameraPose = get_pose(yaw_+M_PI,-pitch_,0,pos_);
  bool isInView = isPointInCameraView(point, cameraPose, cameraIntrinsics, imageSize);
  return isInView;

}


// by zj
Eigen::Matrix4d PerceptionUtils::get_pose(double yaw, double pitch, double roll, const Eigen::Vector3d& position)
{

    yaw = yaw;
    pitch = pitch;
    Eigen::AngleAxisd yawAngle(0, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(yaw, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rollAngle(pitch, Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
    Eigen::Matrix4d transformMatrix = Eigen::Matrix4d::Identity();
    transformMatrix.block<3, 3>(0, 0) = q.toRotationMatrix();
    transformMatrix.block<3, 1>(0, 3) = position;
    return transformMatrix;
}



// by zj
bool PerceptionUtils::isPointInCameraView(const Eigen::Vector3d& point, const Eigen::Matrix4d& cameraPose, const Eigen::Matrix3d& cameraIntrinsics, const Eigen::Vector2i& imageSize)
{

    // 将点从世界坐标系变换到相机坐标系
    Eigen::Vector4d pointCamera = cameraPose.inverse() * point.homogeneous();

    // blender2opencv
    pointCamera[1] = -pointCamera[1];
    pointCamera[2] = -pointCamera[2];

    // 将相机坐标系的点变换到像素坐标系
    Eigen::Vector3d pointPixel = cameraIntrinsics * pointCamera.head<3>() / pointCamera(2);

    // cout << "pointPixel = " << pointPixel << endl;
    // 判断像素坐标是否在图像范围内
    if (pointPixel(0) < 0 || pointPixel(0) > imageSize(0) - 1 || pointPixel(1) < 0 || pointPixel(1) > imageSize(1) - 1)
    {
        // cout << "false" << endl;
        return false;
    }
    // cout << "true" << endl;
    return true;
}


}  // namespace fast_planner