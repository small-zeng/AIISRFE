// #include <active_perception/perception_utils.h>

// #include <pcl/filters/voxel_grid.h>

// using namespace std;

// double fx = 288;
// double fy = 288;
// double cx = 192;
// double cy = 192;


// // by zj
// Eigen::Matrix4d get_pose(double yaw, double pitch, double roll, const Eigen::Vector3d& position)
// {

//     yaw = yaw;
//     pitch = pitch;
//     Eigen::AngleAxisd yawAngle(0, Eigen::Vector3d::UnitZ());
//     Eigen::AngleAxisd pitchAngle(yaw, Eigen::Vector3d::UnitY());
//     Eigen::AngleAxisd rollAngle(pitch, Eigen::Vector3d::UnitX());
//     Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
//     Eigen::Matrix4d transformMatrix = Eigen::Matrix4d::Identity();
//     transformMatrix.block<3, 3>(0, 0) = q.toRotationMatrix();
//     transformMatrix.block<3, 1>(0, 3) = position;
//     return transformMatrix;
// }


// // by zj
// bool isPointInCameraView(const Eigen::Vector3d& point, const Eigen::Matrix4d& cameraPose, const Eigen::Matrix3d& cameraIntrinsics, const Eigen::Vector2i& imageSize)
// {

//     // 将点从世界坐标系变换到相机坐标系
//     Eigen::Vector4d pointCamera = cameraPose.inverse() * point.homogeneous();

//     // blender2opencv
//     pointCamera[1] = -pointCamera[1];
//     pointCamera[2] = -pointCamera[2];

//     // 将相机坐标系的点变换到像素坐标系
//     Eigen::Vector3d pointPixel = cameraIntrinsics * pointCamera.head<3>() / pointCamera(2);

//     cout << "pointPixel = " << pointPixel.transpose() << endl;
//     // 判断像素坐标是否在图像范围内
//     if (pointPixel(0) < 0 || pointPixel(0) > imageSize(0) - 1 || pointPixel(1) < 0 || pointPixel(1) > imageSize(1) - 1)
//     {
//         // cout << "false" << endl;
//         return false;
//     }
//     // cout << "true" << endl;
//     return true;
// }





// int main(int argc, char** argv) {


//   ros::init(argc, argv, "test");
//   ros::NodeHandle nh("~");

//   Eigen::Vector3d point(1.9062,1.33678,1.62934);
  
//   // Eigen::Vector3d pos(1.9062,  2.08678, 0.330301);
//   // double yaw = -0.0613792;
//   // double pitch = 0.507347;

//   Eigen::Vector3d pos(1.9062,   1.65513, -0.176142);
//   double yaw = -0.0340935;
//   double pitch = 0.150154;



//   cout << pos << endl;
//   Eigen::Matrix4d cam_pose = get_pose(yaw+M_PI/2.0*2,-pitch,0,pos);
//   cout << cam_pose << endl;
 
//   Eigen::Matrix3d cameraIntrinsics;
//   Eigen::Vector2i imageSize;
//   cameraIntrinsics << fx, 0.0, cx,
//                     0.0, fy, cy,
//                     0.0, 0.0, 1.0;
//   imageSize << 2*cx,2*cy;

//   bool isInView = isPointInCameraView(point,cam_pose,cameraIntrinsics,imageSize);
//   cout << "isInView = " << isInView << endl;





//   return 0;
// }



#include <ros/ros.h> 
#include <ros/console.h> 
#include <nav_msgs/Path.h> 
#include <std_msgs/String.h> 
#include <geometry_msgs/Quaternion.h> 
#include <geometry_msgs/PoseStamped.h> 
#include <tf/transform_broadcaster.h> 
#include <tf/tf.h> 
#include <httplib.h>
using namespace std;

void get_picture() {
    httplib::Server svr;

    svr.Post("/uploadpicture/", [](const httplib::Request &req, httplib::Response &res) {
		// // 打印请求头
		// std::cout << "Request Headers:\n";
		// for (const auto& header : req.headers) {
		// 	std::cout << header.first << ": " << header.second << std::endl;
		// }


        cout << "is_multipart_form_data() = " << req.is_multipart_form_data() << endl;

        // Process each file in the MultipartFormDataItems
        for (const auto &file : req.files) {
            const std::string &name = file.first;
            const httplib::MultipartFormData &data = file.second;

            const std::string &filename = data.filename;
            const std::string &content_type = data.content_type;
            const std::string &content = data.content;

			cout << filename << endl;
        }




        if (req.has_file("pose")) {
            cout << "get picture" << endl;
            res.set_content("OK", "text/plain");
        }
        else {
            cout << "get picture error!!!!" << endl;
            res.set_content("Bad Request", "text/plain");
        }
       cout << "*******************" << endl;


	});

      svr.listen("10.181.245.129", 7100);
     
    
}



main (int argc, char **argv) 
{ 
	ros::init (argc, argv, "showpath"); 

	ros::NodeHandle ph; 
	ros::Publisher path_pub = ph.advertise<nav_msgs::Path>("/planning_vis/plan_traj",1, true); 

	ros::Time current_time, last_time; 
	current_time = ros::Time::now(); 
	last_time = ros::Time::now(); 

	nav_msgs::Path path; 
	//nav_msgs::Path path; 
	path.header.stamp=current_time; 
	path.header.frame_id="world"; 

	double x = 0.0; 
	double y = 0.0; 
	double th = 0.0; 
	double vx = 0.1; 
	double vy = -0.1; 
	double vth = 0.1; //2运动的三自由度量


	std::thread get_picture_thread(get_picture);
	get_picture_thread.detach();



	ros::Rate loop_rate(1); 
	while (ros::ok()) 
	{ 
		current_time = ros::Time::now(); 
		//compute odometry in a typical way given the velocities of the robot 
		double dt = (current_time - last_time).toSec(); 
		double delta_x = (vx * cos(th) - vy * sin(th)) * dt; 
		double delta_y = (vx * sin(th) + vy * cos(th)) * dt; 
		double delta_th = vth * dt; 

		x += delta_x; 
		y += delta_y; 
		th += delta_th; 

		geometry_msgs::PoseStamped this_pose_stamped; 
		this_pose_stamped.pose.position.x = x; 
		this_pose_stamped.pose.position.y = 1.0; 
        this_pose_stamped.pose.position.z = y;

		geometry_msgs::Quaternion goal_quat = tf::createQuaternionMsgFromYaw(th); 
		this_pose_stamped.pose.orientation.x = goal_quat.x; 
		this_pose_stamped.pose.orientation.y = goal_quat.y; 
		this_pose_stamped.pose.orientation.z = goal_quat.z; 
		this_pose_stamped.pose.orientation.w = goal_quat.w; 

		this_pose_stamped.header.stamp=current_time; 
		this_pose_stamped.header.frame_id="world"; 
		path.poses.push_back(this_pose_stamped); 

		path_pub.publish(path); 
		ros::spinOnce();  // check for incoming messages 

		last_time = current_time; 
		loop_rate.sleep(); 
	} 
	return 0; 
}
