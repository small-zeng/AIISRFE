#include <plan_manage/planner_manager.h>
#include <exploration_manager/fast_exploration_manager.h>
#include <traj_utils/planning_visualization.h>

// #include <exploration_manager/fast_exploration_fsm.h>
#include <generated/fast_exploration_fsm.h>  
#include <exploration_manager/expl_data.h>
#include <plan_env/edt_environment.h>
#include <plan_env/sdf_map.h>
#include <active_perception/surface_finder.h>


#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_transport/image_transport.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include <httplib.h>
// #include <exploration_manager/json.hpp>
#include <sys/stat.h>
#include <sys/types.h> 
#include "visualization_msgs/Marker.h"
#include <visualization_msgs/MarkerArray.h>
#include <jsoncpp/json/json.h>
#include <dirent.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <random>
#include <active_perception/graph_node.h>
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

using Eigen::Vector4d;
using namespace cv;
using namespace std;
using namespace Eigen;



string imgs_base_dir ;  // images path
cv::Mat depth_mat1;
cv::Mat depth_mat;
vector<cv::Mat> depth_arr;
vector<MatrixXd> pose_arr;
int NUM = 0;
int n_index = 0;
double theta = 0.;
int img_index = -1;
vector<int> data_indexs;
int surf_cnt = 0;


Matrix4d unity2blender(Matrix4d & pose){
  MatrixXd T(4,4);
  T << -1,0,0,0,
        0,1,0,0,
        0,0,1,0,
        0,0,0,1;
  pose = T*pose;
  return pose;

}
//a 行  
MatrixXd ReadData(istream & data, int a, int b)
{
	MatrixXd m_matrix(a, b);
	VectorXd hang(a);
	for (int j = 0; j < a; j++)//共a 行
	{
		for (int i = 0; i < b; i++)//共b 列 组成一行
		{
			data >> hang(i);
		}
		m_matrix.row(j) = hang;
	}
	return m_matrix;
}

int readFileList(string basePath,vector<string> &files)
{
    DIR *dir;
    struct dirent *ptr;
 

    if ((dir=opendir(basePath.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)    ///file
        {
            string a = ptr->d_name;
            int pe = a.find_last_of(".");
            string pic_name = a.substr(pe + 1);
            if (pic_name=="txt") //若想获取其他类型文件只需修改jpg为对应后缀
            {
                // string tmpname = basePath + "/" + ptr->d_name;
                string base = "";
                string tmpname = base + ptr->d_name;
                files.push_back(tmpname);
                //name.push_back(ptr->d_name)
            }
        }
        else if(ptr->d_type == 4)    ///dir
        {
            string base = basePath + "/" + ptr->d_name;
            readFileList(base,files);
        }
    }
    closedir(dir);
    return 1;
}



namespace fast_planner {
//planner result
plannerResult planner_res_;

void FastExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);
  fd_.reset(new FSMData);

  /* Initialize main modules */
  expl_manager_.reset(new FastExplorationManager);
  expl_manager_->initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  /*  Fsm param  */
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param<string>("fsm/imgs_mkdir", imgs_base_dir, "");
  nh.param<string>("fsm/local_ip", expl_manager_->ed_->local_ip_, "");

  planner_manager_ = expl_manager_->planner_manager_;
  state_ = EXPL_STATE::WAIT_TRIGGER;
  fd_->have_odom_ = true;
  fd_->state_str_ = { "INIT", "WAIT_TRIGGER", "PLAN_TRAJ", "PUB_TRAJ", "EXEC_TRAJ", "FINISH", "TEST","RUN" };
  fd_->trigger_ = false;
  expl_manager_->frontier_call_flag = true;
  expl_manager_->ed_->is_exp = true;    // by zj for test, set  true for exploration tasks, false for reconstruction tasks
  expl_manager_->ed_->mode_num = 2;     // by zj for test, set 2 for two tasks, 1 for only reconstruction tasks
  expl_manager_->ed_->is_only_exp = false;     // by zj for test, set true for only exploration tasks, false for two tasks              
  expl_manager_->ed_->exp_mode_num = 0; // 
  expl_manager_->ed_->rec_mode_num = 0; //
  ROS_WARN("[FSM INIT]Method index = %d", METHOD_INDEX);
  if (METHOD_INDEX == 0) expl_manager_->ed_->is_3D = false;
  else expl_manager_->ed_->is_3D = true;

  expl_manager_->ed_->all_path = vector<Eigen::Vector4d>();
  ROS_WARN("expl_manager_->ed_->is_only_exp = %s",expl_manager_->ed_->is_only_exp ? "true" : "false");

  // Metric files init
  fs::path fullPath(imgs_base_dir);
  fs::path parentPath = fullPath.parent_path();
  expl_manager_->ed_->metric_path = parentPath.string() + "/metric";
  createFolder(expl_manager_->ed_->metric_path);
  std::string pathFilename = expl_manager_->ed_->metric_path + "/path.txt";
  std::string viewsFilename = expl_manager_->ed_->metric_path + "/views.txt";
  std::string timeFilename = expl_manager_->ed_->metric_path + "/time.txt";
  // removeIfExists(pathFilename);
  // removeIfExists(viewsFilename);
  // removeIfExists(timeFilename);
  createEmptyFileIfNotExists(pathFilename);
  createEmptyFileIfNotExists(viewsFilename);
  createEmptyFileIfNotExists(timeFilename);
  expl_manager_->ed_->planning_num = 0;


  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.1), &FastExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &FastExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.5), &FastExplorationFSM::frontierCallback, this);
  send_data_timer_ = nh.createTimer(ros::Duration(1.0), &FastExplorationFSM::send_data_periodicCallback,this);
  // sample_test_views_timer_ = nh.createTimer(ros::Duration(1.0), &FastExplorationFSM::sample_test_views_periodicCallback,this);



  trigger_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1, &FastExplorationFSM::triggerCallback, this);
  odom_sub_ = nh.subscribe("/odom_world", 1, &FastExplorationFSM::odometryCallback, this);

  odom_pub_ = nh.advertise<nav_msgs::Odometry>("/odom_world", 1000);
  replan_pub_ = nh.advertise<std_msgs::Empty>("/planning/replan", 10);
  new_pub_ = nh.advertise<std_msgs::Empty>("/planning/new", 10);
  bspline_pub_ = nh.advertise<bspline::Bspline>("/planning/bspline", 10);
  expl_manager_->ed_->all_traj_pub = nh.advertise<visualization_msgs::Marker>("/all/plan_traj", 10);

        
  /* init param */
  depth_pub = nh.advertise<sensor_msgs::Image>("/pcl_render_node/depth", 1000);
  pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/pcl_render_node/sensor_pose", 1000);
  getdata_timer_ = nh.createTimer(ros::Duration(0.005), &FastExplorationFSM::getdataCallback, this); // 0.05 - > 0.005 by zj
  std::thread is_finish_thread(&FastExplorationFSM::is_finish,this);
  is_finish_thread.detach();
  std::thread read_data_thread(&FastExplorationFSM::read_data,this);
  read_data_thread.detach();

  ros::Duration(1.0).sleep();

  // // Gibson Rosser
  // Eigen::Vector3d start_pos(1,1,-4);

  // // Gibson Convoy
  // Eigen::Vector3d start_pos(0,1,0);

  // Room childroom
  Eigen::Vector3d start_pos(0,1.5,-3.5);

    // Room childroom
  // Eigen::Vector3d start_pos(0,1,-5);

  for (int i=0; i<2;i++){
      auto location = start_pos; 
      thread send_NBV_thread(&FastExplorationManager::send_NBV, expl_manager_,location,0,-5.0/180*M_PI+i*5.0/180*M_PI);  // i*90.0/180*M_PI
      send_NBV_thread.detach();
      ros::Duration(3.0).sleep();
  } 

  fd_->start_pt_ = start_pos;
  fd_->start_vel_.setZero();
  fd_->start_acc_.setZero();
  fd_->start_dir_ = Eigen::Vector3d(0.0,0.0,0.0);

  // std::thread demo_traj_thread(&FastExplorationFSM::demo_traj,this);
  // demo_traj_thread.detach();

}



void FastExplorationFSM::demo_traj(){
    // 读取规划的数据
    for(int n=0; n<259; n++){
        string depth_dir = imgs_base_dir + "/"  + to_string(n)+"_depth.png";
        depth_mat1 = imread(depth_dir,cv::IMREAD_ANYDEPTH);
        depth_mat1.convertTo(depth_mat,CV_32FC1);
        depth_mat = depth_mat /65535.0*10.0;
        addDepthNoise(depth_mat, 10.0); 
        cout << "depth mat size = " << depth_mat.rows << ", " << depth_mat.cols << endl;
        depth_arr.push_back(depth_mat.clone());
        
        // read pose
        string pose_dir = imgs_base_dir + "/"  + to_string(n)+"_pose.txt";
        ifstream in(pose_dir, ios::in);
        if (!in)
        {
          cerr<<"位姿文件不存在"<<endl;
        }
        Matrix4d pose_mat = ReadData(in, 4, 4);
        // Blender
        unity2blender(pose_mat);
        pose_arr.push_back(pose_mat);

    }

    // // 发布初始状态
    // for (int i=0; i<5; i++){
    //   img_index ++;
    //   thread pubeImage_thread(&FastExplorationFSM::publishImage, this);
    //   pubeImage_thread.detach();
    // }

    // 读取存入的 metric文件
    vector<Vector3d> path_pos;
    vector<double> path_pitchs, path_yaws;
    path_pos.push_back(Eigen::Vector3d(0,1.5,-2));
    path_pitchs.push_back(0.0);
    path_yaws.push_back(0.0);
    expl_manager_->frontier_finder_->drawPathViews(path_pos,path_pitchs,path_yaws);
    string path_dir = imgs_base_dir + "/../metric/path.txt";
    string views_dir = imgs_base_dir + "/../metric/views.txt";
    cout << path_dir << endl;
    vector<Eigen::MatrixXf> all_traj_matrix = readMetric(path_dir,4);
    vector<Eigen::MatrixXf> all_views_matrix = readMetric(views_dir,5);
    cout << "all_traj size = " << all_traj_matrix.size() << endl;
    cout << "all_views size = " << all_views_matrix.size() << endl;
    vector<Vector4d> all_traj;
    for (int k=0; k<all_traj_matrix.size(); k++){
        Eigen::MatrixXf path_matrix = all_traj_matrix.at(k);
        Eigen::MatrixXf views_matrix = all_views_matrix.at(k);
        for (int i = 0; i < path_matrix.rows(); ++i) {
            Vector4d traj_point;
            traj_point << path_matrix(i, 0), path_matrix(i, 1), path_matrix(i, 2), path_matrix(i, 3); 
            // cout << traj_point.transpose() << endl;
            all_traj.push_back(traj_point);
        }
        expl_manager_->drawAllTraj(all_traj);
        cout << "views_matrix size = " << views_matrix.size() << endl;;
        for (int j = 0; j < views_matrix.rows(); ++j) {
              path_pos.clear();
              path_pitchs.clear();
              path_yaws.clear();
              expl_manager_->frontier_finder_->drawPathViews(path_pos,path_pitchs,path_yaws);
              path_pos.push_back(Eigen::Vector3d(views_matrix(j, 0), views_matrix(j, 1), views_matrix(j, 2))); 
              path_pitchs.push_back(views_matrix(j, 3));
              path_yaws.push_back(views_matrix(j, 4));
              expl_manager_->frontier_finder_->drawPathViews(path_pos,path_pitchs,path_yaws);
              img_index ++;
              thread pubeImage_thread(&FastExplorationFSM::publishImage, this);
              pubeImage_thread.detach();
              ros::Duration(0.15).sleep();
        }
    }
  

}



void FastExplorationFSM::publishImage(){
   // // online 
    if (img_index < 0)
    // if (img_index < 0 || img_index > 100) // by zj for test
      return;
    n_index = img_index;
  
    // cout << "n_index = " << n_index << " , depth size = " << depth_arr.size() << " , pose size = " << pose_arr.size() << endl;

    cv_bridge::CvImage out_msg;
    ros::Time t = ros::Time::now();
    out_msg.header.stamp = t;
    out_msg.header.frame_id = "world";
    out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    out_msg.image = depth_arr.at(n_index).clone();
    depth_pub.publish(out_msg.toImageMsg());


    // cout<<"pub cam pose"
    geometry_msgs::PoseStamped camera_pose;
    camera_pose.header.stamp = t;
    // camera_pose.header = odom_.header;
    camera_pose.header.frame_id = "/map";
    camera_pose.pose.position.x = pose_arr.at(n_index)(0,3);
    camera_pose.pose.position.y = pose_arr.at(n_index)(1,3);
    camera_pose.pose.position.z = pose_arr.at(n_index)(2,3);

    Matrix3d R = pose_arr.at(n_index).block(0,0,3,3);
    Quaterniond Q(R);
    // cout << "R (before) = " << R << endl;
    // cout << typeid(Q) << endl;
    // cout << "Q = " << Q.coeffs() << endl;
    camera_pose.pose.orientation.w = Q.coeffs()(3);
    camera_pose.pose.orientation.x = Q.coeffs()(0);
    camera_pose.pose.orientation.y = Q.coeffs()(1);
    camera_pose.pose.orientation.z = Q.coeffs()(2);
    pose_pub.publish(camera_pose);

    nav_msgs::Odometry odom;
    odom.header.stamp = t;
    odom.header.frame_id = "/odom";

    odom.pose.pose.position.x = camera_pose.pose.position.x;
    odom.pose.pose.position.y = camera_pose.pose.position.y;
    odom.pose.pose.position.z = camera_pose.pose.position.z;
    odom.pose.pose.orientation = camera_pose.pose.orientation;
    odom_pub_.publish(odom);
    
    // cout << "publishImage: " << img_index << endl;
    // cout << "zj is cool" << endl;


}

void FastExplorationFSM::FSMCallback(const ros::TimerEvent& e) {
  ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << fd_->state_str_[int(state_)]);
  // cout << "state = " << fd_->state_str_[int(state_)] << endl;

  switch (state_) {
    case TEST: {
      // Do nothing but wait for trigger
      ROS_WARN_THROTTLE(1.0, "TEST");
      break;
    }

    case INIT: {
      // Wait for odometry ready
      if (!fd_->have_odom_) {
        ROS_WARN_THROTTLE(1.0, "no odom.");
        return;
      }
      // Go to wait trigger when odom is ok
      transitState(WAIT_TRIGGER, "FSM");
      break;
    }

    case WAIT_TRIGGER: {
      // Do nothing but wait for trigger
      // ROS_WARN_THROTTLE(1.0, "wait for trigger.");
      break;
    }

    case FINISH: {
      ROS_INFO_THROTTLE(1.0, "finish exploration.");
      break;
    }

    case PLAN_TRAJ: {

      ROS_WARN("Method index = %d", METHOD_INDEX);
      expl_manager_->ed_->T_task = 0.0;
      expl_manager_->ed_->T_atsp = 0.0;
      expl_manager_->ed_->T_switch = 0.0;
      expl_manager_->ed_->T_sp = 0.0;

      /*
        Our method 
        设置为 mode switching
      */
      if (METHOD_INDEX == 6){
          // Mode switching
          ros::Time T_0 = ros::Time::now();
          ROS_WARN("expl_manager_->ed_->rec_mode_num = %d",expl_manager_->ed_->rec_mode_num);
          auto ft = expl_manager_->frontier_finder_;
          expl_manager_->frontier_call_flag = false;
          ft->searchFrontiers();
          ft->computeFrontiersToVisit(expl_manager_->ed_->is_3D);
          ROS_WARN("[EXP] num of frontiers = %d",ft->frontiers_.size());
          int k = 0;
          vector<Vector3d> path;
          int valid_frontier_num = 0;
          for (auto frontier : ft->frontiers_){
            // double dist = (frontier.viewpoints_[0].pos_ - fd_->start_pt_).norm();
            double dist = ViewNode::searchPath(fd_->start_pt_, frontier.viewpoints_[0].pos_, path) ;
            cout << k << " | dist = " << dist << endl;
            if (dist < 1.0){
              valid_frontier_num += 1;
            }
            k++;
          }
          double valid_ratio = static_cast<double>(valid_frontier_num)/ft->frontiers_.size();
          ROS_WARN("[EXP] ratio of valid frontiers = %f, valid_frontier_num = %d",valid_ratio, valid_frontier_num);
          if (expl_manager_->ed_->mode_num == 1){
            expl_manager_->ed_->is_exp = false;
          }
          else if (expl_manager_->ed_->rec_mode_num < 2  && (valid_ratio < 0.15 || valid_frontier_num < 2) && img_index > 10){
            ROS_WARN("[EXP] No enough valid frontiers");
            expl_manager_->ed_->is_exp = false;
            
            if (expl_manager_->ed_->mode_num == 2) 
                expl_manager_->ed_->rec_mode_num += 1;
            else 
                expl_manager_->ed_->rec_mode_num = 0;
          }
          else{
            expl_manager_->ed_->is_exp = true;
            expl_manager_->ed_->rec_mode_num = 0;
          
          }
          expl_manager_->ed_->T_switch = (ros::Time::now()-T_0).toSec();
      }

      /*
        Fuel / Fuel*  
        设置为 Exploration tasks
      */
      if (METHOD_INDEX == 0 || METHOD_INDEX == 1){
          expl_manager_->ed_->is_exp = true;
          expl_manager_->ed_->T_switch = 0.0; 
      }
     
      /*
        EVPP NBV / VPP TRO / only reconstruction tasks / noswitch 
        设置为 Reconstruction tasks
      */
      if (METHOD_INDEX == 2 || METHOD_INDEX == 3 || METHOD_INDEX == 4 || METHOD_INDEX == 5){
          expl_manager_->ed_->is_exp = false; 
          expl_manager_->ed_->T_switch = 0.0;  
      }

      // 打印当前模式
      if (expl_manager_->ed_->is_exp){
        ROS_WARN("*************Exploration tasks mode*************");
        ROS_WARN("Mode num = %d",expl_manager_->ed_->mode_num);
      }else{
        ROS_WARN("*************Reconstruction tasks mode*************");
        ROS_WARN("Mode num = %d",expl_manager_->ed_->mode_num);
      }

      ros::Time T_s = ros::Time::now();
      int res = callExplorationPlanner();
      expl_manager_->frontier_finder_->drawFirstViews();
      expl_manager_->ed_->T_sp = (ros::Time::now() - T_s).toSec();
      if (res == SUCCEED){
          save_metric();
      }  
      expl_manager_->frontier_call_flag = true;
      if(expl_manager_->ed_->is_exp == true){
          cout << "res = " << res << endl;
          if (res == SUCCEED) {
            expl_manager_->frontier_finder_->drawTraj(expl_manager_->ed_->exp_path); // by zj for test
            thread vis_thread(&FastExplorationFSM::visualize, this);
            vis_thread.detach();
            transitState(RUN, "FSM");
          } else if (res == NO_FRONTIER) {
            transitState(FINISH, "FSM");
          } else if (res == FAIL) {
            ROS_WARN("plan fail");
            transitState(PLAN_TRAJ, "FSM");
          }
        
      }
      else {
          if (res == SUCCEED){
            transitState(RUN, "FSM");
          }
          else{
            ROS_WARN("plan fail");
            transitState(PLAN_TRAJ, "FSM");
          }

          

      }

      expl_manager_->drawAllTraj(expl_manager_->ed_->all_path);
      

      break;
    }

    case RUN: {


      break;
    }


  
  }
}

void FastExplorationFSM::send_data_periodicCallback(const ros::TimerEvent& event){

  vector<Vector3d>  path_pos;
  vector<double>  path_pitchs, path_yaws;
  // ROS_WARN("[send_data_periodicCallback] planner_res_.res_pts_.size() = %d", planner_res_.res_pts_.size());
  if (planner_res_.res_pts_.size() > 0){
      thread send_NBV_thread(&FastExplorationManager::send_NBV, expl_manager_,planner_res_.res_pts_[0],
                            planner_res_.res_pitchs_[0],planner_res_.res_yaws_[0]);
      send_NBV_thread.detach();
      // draw path views
      path_pos.push_back(planner_res_.res_pts_[0]);
      path_pitchs.push_back(planner_res_.res_pitchs_[0]);
      path_yaws.push_back(planner_res_.res_yaws_[0]);
      expl_manager_->frontier_finder_->drawPathViews(path_pos,path_pitchs,path_yaws);
      path_pos.clear();
      path_pitchs.clear();
      path_yaws.clear();
      planner_res_.res_pts_.erase(planner_res_.res_pts_.begin());
      planner_res_.res_pitchs_.erase(planner_res_.res_pitchs_.begin());
      planner_res_.res_yaws_.erase(planner_res_.res_yaws_.begin());

  }
  else{
    ROS_INFO("All views sended ");
    // transitState to PLAN_TRAJ when all views received


    if(data_indexs.size()>2 && data_indexs.back() == img_index ){
        ROS_WARN("All views sended ");
        if (expl_manager_->ed_->is_exp == true){
          ros::Duration(3.0).sleep();
          transitState(PLAN_TRAJ, "FSM");
          
        }
        else{
          ros::Duration(3.0).sleep();
          transitState(PLAN_TRAJ, "FSM");
      }
    

    }
  }


}


int FastExplorationFSM::callExplorationPlanner() {
  
  cout << "start = " << fd_->start_pt_.transpose() << endl;
  int res;
  Eigen::Vector4d pos;
  expl_manager_->ed_->single_path.clear();
  expl_manager_->ed_->path_length = 0.0;
  expl_manager_->ed_->single_views.clear();
  if (expl_manager_->ed_->is_exp){
      planner_res_ = expl_manager_->planExploreMotion_EXP(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_, fd_->start_dir_);
      ROS_WARN("[EXP] num of views = %d",planner_res_.res_pts_.size());
      res = planner_res_.res_state_;
      if (res == SUCCEED) {
          fd_->start_pt_ = planner_res_.res_pts_.back();
          fd_->start_vel_.setZero();
          fd_->start_acc_.setZero();
          fd_->start_dir_ = Eigen::Vector3d(planner_res_.res_yaws_.back(),planner_res_.res_pitchs_.back(),0.0);
          cout << "goal = " << planner_res_.res_pts_.back().transpose() << endl;

          // compute path length, get single path, planned views
          pos << expl_manager_->ed_->exp_path[0][0], expl_manager_->ed_->exp_path[0][1],
                                expl_manager_->ed_->exp_path[0][2], 1;
          expl_manager_->ed_->single_path.push_back(pos);
          for (int i = 0; i < expl_manager_->ed_->exp_path.size()-1; i++){
            expl_manager_->ed_->path_length += (expl_manager_->ed_->exp_path[i+1] - expl_manager_->ed_->exp_path[i]).norm();
            pos << expl_manager_->ed_->exp_path[i+1][0], expl_manager_->ed_->exp_path[i+1][1],
                      expl_manager_->ed_->exp_path[i+1][2], 1;
            expl_manager_->ed_->single_path.push_back(pos);
          }
          
          for (int i=0; i < planner_res_.res_pts_.size(); i++ ){
            vector<double> view;
            view.push_back(planner_res_.res_pts_[i][0]);
            view.push_back(planner_res_.res_pts_[i][1]);
            view.push_back(planner_res_.res_pts_[i][2]);
            view.push_back(planner_res_.res_pitchs_[i]);
            view.push_back(planner_res_.res_yaws_[i]);
            expl_manager_->ed_->single_views.push_back(view);
          }


      }

      if (res == NO_FRONTIER && expl_manager_->ed_->is_only_exp == false){
        expl_manager_->ed_->is_exp = false;
        expl_manager_->ed_->mode_num = 1;
        ROS_WARN("Change to reconstruction tasks");
      }
  }

  else{
     planner_res_ = expl_manager_->planExploreMotion_REC(fd_->start_pt_, fd_->start_vel_, fd_->start_acc_, fd_->start_dir_);
     ROS_WARN("[REC] num of views = %d",planner_res_.res_pts_.size());
     res = planner_res_.res_state_;

     if (res == SUCCEED ) {
          fd_->start_pt_ = planner_res_.res_pts_.back();
          fd_->start_vel_.setZero();
          fd_->start_acc_.setZero();
          fd_->start_dir_ = Eigen::Vector3d(planner_res_.res_yaws_.back(),planner_res_.res_pitchs_.back(),0.0);
          cout << "goal = " << planner_res_.res_pts_.back().transpose() << endl;

          // compute path length, get single path, planned views
          pos << expl_manager_->ed_->surf_path[0][0], expl_manager_->ed_->surf_path[0][1],
                                expl_manager_->ed_->surf_path[0][2], 0;
          expl_manager_->ed_->single_path.push_back(pos);
          for (int i = 0; i < expl_manager_->ed_->surf_path.size()-1; i++){
            expl_manager_->ed_->path_length += (expl_manager_->ed_->surf_path[i+1] - expl_manager_->ed_->surf_path[i]).norm();
            pos << expl_manager_->ed_->surf_path[i+1][0], expl_manager_->ed_->surf_path[i+1][1],
                      expl_manager_->ed_->surf_path[i+1][2], 0;
            expl_manager_->ed_->single_path.push_back(pos);
          }

          for (int i=0; i < planner_res_.res_pts_.size(); i++ ){
            vector<double> view;
            view.push_back(planner_res_.res_pts_[i][0]);
            view.push_back(planner_res_.res_pts_[i][1]);
            view.push_back(planner_res_.res_pts_[i][2]);
            view.push_back(planner_res_.res_pitchs_[i]);
            view.push_back(planner_res_.res_yaws_[i]);
            expl_manager_->ed_->single_views.push_back(view);
          }



        
      }



  }

  
  cout << "******** Finish view path planning ********" << endl;

  return res;
}


void FastExplorationFSM::visualize() {
  auto info = &planner_manager_->local_data_;
  auto plan_data = &planner_manager_->plan_data_;
  auto ed_ptr = expl_manager_->ed_;
  auto ft = expl_manager_->frontier_finder_;

  // Draw views
  // ft->drawFirstViews();
  // ft->drawAllViews();

  // Draw frontier
  static int last_ftr_num = 0;
  for (int i = 0; i < ed_ptr->frontiers_.size(); ++i) {
    visualization_->drawCubes(ed_ptr->frontiers_[i], 0.1,
                              visualization_->getColor(double(i) / ed_ptr->frontiers_.size(), 0.4),
                              "frontier", i, 4);
    // visualization_->drawBox(ed_ptr->frontier_boxes_[i].first, ed_ptr->frontier_boxes_[i].second,
    //                         Vector4d(0.5, 0, 1, 0.3), "frontier_boxes", i, 4);
  }
  for (int i = ed_ptr->frontiers_.size(); i < last_ftr_num; ++i) {
    visualization_->drawCubes({}, 0.1, Vector4d(0, 0, 0, 1), "frontier", i, 4);
    // visualization_->drawBox(Vector3d(0, 0, 0), Vector3d(0, 0, 0), Vector4d(1, 0, 0, 0.3),
    // "frontier_boxes", i, 4);
  }
  last_ftr_num = ed_ptr->frontiers_.size();

}

void FastExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  static int delay = 0;
  if (++delay < 5) return;

  if (state_ == WAIT_TRIGGER || state_ == FINISH || expl_manager_->frontier_call_flag == true) {
    cout << "come into frontierCallback" << endl;
    auto ft = expl_manager_->frontier_finder_;
    auto ed = expl_manager_->ed_;
    ros::Time t0 = ros::Time::now();

    ft->searchFrontiers(expl_manager_->ed_->is_3D);
    // cout << "searchFrontiers() time: " << (ros::Time::now()-t0).toSec() << endl;
    
    ft->computeFrontiersToVisit(expl_manager_->ed_->is_3D);
    // cout << "computeFrontiersToVisit() time: " << (ros::Time::now()-t0).toSec() << endl;
    
    // ft->updateFrontierCostMatrix();
    // cout << "updateFrontierCostMatrix() time: " << (ros::Time::now()-t0).toSec() << endl;
    

    ft->getFrontiers(ed->frontiers_);
    ft->getFrontierBoxes(ed->frontier_boxes_);


    // //Draw frontier and bounding box
    for (int i = 0; i < ed->frontiers_.size(); ++i) {
      // if (i!=2) continue;
      visualization_->drawCubes(ed->frontiers_[i], 0.1,
                                visualization_->getColor(double(i) / ed->frontiers_.size(), 0.4),
                                "frontier", i, 4);
    }


  }



}

void FastExplorationFSM::triggerCallback(const nav_msgs::PathConstPtr& msg) {
  // if (msg->poses[0].pose.position.z < -0.1) return;
  // if (state_ != WAIT_TRIGGER) return;
  fd_->trigger_ = true;
  cout << "Triggered!" << endl;
  transitState(PLAN_TRAJ, "triggerCallback");
}

void FastExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (state_ == EXPL_STATE::EXEC_TRAJ) {
    // Check safety and trigger replan if necessary
    double dist;
    bool safe = planner_manager_->checkTrajCollision(dist);
    if (!safe) {
      ROS_WARN("Replan: collision detected==================================");
      transitState(PLAN_TRAJ, "safetyCallback");
    }
  }
}

void FastExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {

  // cout << "odometryCallback" << endl;

  fd_->odom_pos_(0) = msg->pose.pose.position.x;
  fd_->odom_pos_(1) = msg->pose.pose.position.y;
  fd_->odom_pos_(2) = msg->pose.pose.position.z;

  fd_->odom_vel_(0) = msg->twist.twist.linear.x;
  fd_->odom_vel_(1) = msg->twist.twist.linear.y;
  fd_->odom_vel_(2) = msg->twist.twist.linear.z;

  fd_->odom_orient_.w() = msg->pose.pose.orientation.w;
  fd_->odom_orient_.x() = msg->pose.pose.orientation.x;
  fd_->odom_orient_.y() = msg->pose.pose.orientation.y;
  fd_->odom_orient_.z() = msg->pose.pose.orientation.z;

  Eigen::Vector3d rot_x = fd_->odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  fd_->odom_yaw_ = atan2(rot_x(1), rot_x(0));

  fd_->have_odom_ = true;
}


void FastExplorationFSM::transitState(EXPL_STATE new_state, string pos_call) {
  int pre_s = int(state_);
  state_ = new_state;
  cout << "[" + pos_call + "]: from " + fd_->state_str_[pre_s] + " to " + fd_->state_str_[int(new_state)]
       << endl;
}

// offline test
void FastExplorationFSM::getdataCallback(const ros::TimerEvent& e) {

  // cout << "getdataCallback" << endl;
  vector<string> temp;//文件路径
  vector<int> indexs;
  readFileList(imgs_base_dir, temp);
  if(temp.size() > 0)
  {
    for (int i = 0; i<temp.size(); i++) {
        vector<string> vStr;
        boost::split( vStr, temp[i], boost::is_any_of( "_" ), boost::token_compress_on );
        // cout << vStr.front() << endl;
        indexs.push_back(atoi(vStr.front().c_str()));
    }
    sort(indexs.begin(),indexs.end()); //默认从小到大排序
    data_indexs.assign(indexs.begin(),indexs.end()); // 清空并深拷贝
    // cout << "indexs = ";
    // for (int i = 0; i<temp.size(); i++) {
    //     cout << data_indexs[i] << ", ";
    // }
    // cout << endl;
  }

  thread pubeImage_thread(&FastExplorationFSM::publishImage, this);
  pubeImage_thread.detach();


}


// when planner is finished 
void FastExplorationFSM::is_finish(){

    httplib::Server svr;
    cout << "come into  is_finish" << endl;
    svr.Get("/isfinish", [this](const httplib::Request& req, httplib::Response& res) {
        cout << "isfinish " << endl;
        fd_->trigger_ = true;
        cout << "Triggered!  " << "img_index = " << img_index << endl;
        transitState(PLAN_TRAJ, "triggerCallback");
        res.set_content("Hello, World!", "text/plain");
    });

    // svr.listen(expl_manager_->ed_->local_ip_, 7300);
    svr.listen("127.0.0.1", 7300);


}

// read data from indexs
void FastExplorationFSM::read_data(){
    while (true)
    {
        
        // cout << "data_indexs.size() = " << data_indexs.size() << endl;
        if (data_indexs.size()>0 && data_indexs.back() > img_index){

            // cout << "buf_img_index = " << img_index << endl;//输出读取的文本文件数据
            // read depth
            string depth_dir = imgs_base_dir + "/"  + to_string(img_index+1)+"_depth.png";
            depth_mat1 = imread(depth_dir,cv::IMREAD_ANYDEPTH);
            depth_mat1.convertTo(depth_mat,CV_32FC1);
            depth_mat = depth_mat /65535.0*10.0;
            // addDepthNoise(depth_mat, 10.0); 
            // cout << "depth mat size = " << depth_mat.rows << ", " << depth_mat.cols << endl;
            depth_arr.push_back(depth_mat.clone());
            
            // read pose
            string pose_dir = imgs_base_dir + "/"  + to_string(img_index+1)+"_pose.txt";
            ifstream in(pose_dir, ios::in);
            if (!in)
            {
              cerr<<"位姿文件不存在"<<endl;
            }
            Matrix4d pose_mat = ReadData(in, 4, 4);
            // Blender
            unity2blender(pose_mat);
            pose_arr.push_back(pose_mat);
            // cout << "pose = \n" << pose_mat << endl;
            
            img_index += 1;
            ROS_WARN("img_index: %d", img_index);

            thread pubeImage_thread(&FastExplorationFSM::publishImage, this);
            pubeImage_thread.detach();

          }

          ros::Duration(0.3).sleep();

   }
  

}

// add depth noise to depth_mat
void FastExplorationFSM::addDepthNoise(cv::Mat& depth_mat, double far) {
    std::random_device rd;
    std::mt19937 generator(rd());

    for (int i = 0; i < depth_mat.rows; ++i) {
        for (int j = 0; j < depth_mat.cols; ++j) {
            float depth_value = depth_mat.at<float>(i, j);
            float squared_depth = depth_value * depth_value;

            float std_dev = 2.925 * squared_depth + 3.325;

            std::normal_distribution<float> distribution(0.0, std_dev);
            float noise = distribution(generator) / far / 1000;

            depth_mat.at<float>(i, j) += noise;
        }
    }
}

void FastExplorationFSM::removeIfExists(const std::string& filename) {
    std::remove(filename.c_str());
}

void FastExplorationFSM::createEmptyFileIfNotExists(const std::string& filename) {
    std::ifstream fileCheck(filename);
    
    if (!fileCheck.is_open()) {
        // File doesn't exist, create a new empty file
        std::ofstream outputFile(filename);
        if (!outputFile.is_open()) {
            std::cerr << "Failed to create the file." << std::endl;
            return;
        }
        outputFile.close();
    } else {
        // File already exists, do nothing
        fileCheck.close();
    }
}

bool FastExplorationFSM::createFolder(const std::string& folderPath) {
    if (!fs::exists(folderPath)) {
        if (fs::create_directory(folderPath)) {
            std::cout << "Folder created successfully!" << std::endl;
            return true;
        } else {
            std::cerr << "Failed to create folder." << std::endl;
            return false;
        }
    } else {
        std::cout << "Folder already exists." << std::endl;
        return true;
    }
}

void FastExplorationFSM::save_metric(){
    ROS_WARN("mode = %d, planner time = %f, T_switch = %f, path_length = %f", expl_manager_->ed_->is_exp, 
          expl_manager_->ed_->T_sp, expl_manager_->ed_->T_switch, expl_manager_->ed_->path_length);

    expl_manager_->ed_->planning_num ++;
    std::string pathFilename = expl_manager_->ed_->metric_path + "/path.txt";
    std::string viewsFilename = expl_manager_->ed_->metric_path + "/views.txt";
    std::string timeFilename = expl_manager_->ed_->metric_path + "/time.txt";

    // save path
    std::ofstream pathFile_output(pathFilename, std::ios::app);
    pathFile_output << "planned path :planned num =" + to_string(expl_manager_->ed_->planning_num) + "," + 
         "is_exp =" + to_string(expl_manager_->ed_->is_exp) + "," + 
         "path_length =" + to_string(expl_manager_->ed_->path_length) << '\n';
    for (auto& pos : expl_manager_->ed_->single_path) {
        std::string data;
        for (size_t j = 0; j < pos.size(); ++j) {
            data += std::to_string(pos[j]);
            if (j < pos.size() - 1) {
                data += ",";
            }
        }
        pathFile_output << data << '\n';
    }
    pathFile_output.close();

    // save time
    cout << "T_sp = " << expl_manager_->ed_->T_sp << ", T_atsp = " << expl_manager_->ed_->T_atsp << endl;
    expl_manager_->ed_->T_task = expl_manager_->ed_->T_sp - expl_manager_->ed_->T_atsp;
    expl_manager_->ed_->T_sp = expl_manager_->ed_->T_sp + expl_manager_->ed_->T_switch;
    std::ofstream timeFile_output(timeFilename, std::ios::app);
    timeFile_output << "planned time :planned num =" + to_string(expl_manager_->ed_->planning_num) + "," + 
         "is_exp =" + to_string(expl_manager_->ed_->is_exp) + "," + 
         "path_length =" + to_string(expl_manager_->ed_->path_length) << '\n';
    std::string data = to_string(expl_manager_->ed_->T_task) + "," + to_string(expl_manager_->ed_->T_atsp) + "," 
          + to_string(expl_manager_->ed_->T_switch) + "," + to_string(expl_manager_->ed_->T_sp);
    timeFile_output << data << '\n';
    timeFile_output.close();

    // save views
    std::ofstream viewsFile_output(viewsFilename, std::ios::app);
    viewsFile_output << "planned views :planned num =" + to_string(expl_manager_->ed_->planning_num) + "," + 
         "is_exp =" + to_string(expl_manager_->ed_->is_exp) + "," + 
         "path_length =" + to_string(expl_manager_->ed_->path_length) + "," + 
         "views_num =" + to_string(expl_manager_->ed_->single_views.size()) << '\n';
    for (auto& view : expl_manager_->ed_->single_views) {
        std::string data;
        for (size_t j = 0; j < view.size(); ++j) {
            data += std::to_string(view[j]);
            if (j < view.size() - 1) {
                data += ",";
            }
        }
        viewsFile_output << data << '\n';
    }
    viewsFile_output.close();


}

void FastExplorationFSM::sampleTestViews(vector<Vector3d> & sample_pos, vector<double> & sample_pitchs, 
                           vector<double> & sample_yaws, vector<double> & uncers, int N){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distX(-25, 25);
    std::uniform_real_distribution<double> distY(-5, 5);
    std::uniform_real_distribution<double> distZ(-25, 25);
    std::uniform_real_distribution<double> distYAW(-M_PI, M_PI);
    std::uniform_real_distribution<double> distRandom(0.0, 1.0);

    int num_accepted_samples = 0;

    while (num_accepted_samples < N) {
        double x = distX(gen);
        double y = distY(gen);
        double z = distZ(gen);
        Eigen::Vector3d sampled_location(x, y, z);
        double v = distYAW(gen);
        double dy = 1.7 - y;
        double u_center = std::atan2(-dy, 0.4);
        double uncer = distRandom(gen);
        // cout << "sampled_location = " << sampled_location.transpose() << endl;

        if(sampled_location[1] > 2.1 || sampled_location[1]<1.1 ) continue;  // by zj
        // Qualified viewpoint is in bounding box and in safe region
        if (!expl_manager_->frontier_finder_->edt_env_->sdf_map_->isInBox(sampled_location) ||
            expl_manager_->frontier_finder_->edt_env_->sdf_map_->getInflateOccupancy(sampled_location) == 1|| expl_manager_->frontier_finder_->isNearUnknown(sampled_location))
          continue;
        
        double dist = expl_manager_->frontier_finder_->edt_env_->sdf_map_->getDistance(sampled_location);
        // cout << "dist = " << dist << endl;
        if (dist < 0.4) continue;  
         
        sample_pos.push_back(sampled_location);
        sample_pitchs.push_back(u_center);
        sample_yaws.push_back(v);
        uncers.push_back(uncer);

        num_accepted_samples++;


    }

}

void FastExplorationFSM::sample_test_views_periodicCallback(const ros::TimerEvent& event){

  ROS_WARN("[fast_exploration_evpp]come into sample_test_views");
  static int delay = 0;
  if (++delay < 5) return;


  vector<Vector3d> sample_pos;
  vector<double> sample_pitchs, sample_yaws, uncers;
  int N = 240;
  sampleTestViews(sample_pos, sample_pitchs, sample_yaws, uncers, N);
  expl_manager_->surface_finder_->drawSampleViews(sample_pos,sample_pitchs,sample_yaws,uncers);

   // save views
  string testViewsFilename = "/mnt/dataset/zengjing/monosdf_planning/data/Views/childroom.txt";
  std::ofstream testViewsFile_output(testViewsFilename);
  for (int i = 0; i < sample_pos.size(); i++){
    string data = to_string(-sample_pos[i][0]) + " " + to_string(sample_pos[i][1]) + " " + to_string(sample_pos[i][2]) + " "
               + to_string(sample_pitchs[i]) + " " + to_string(-sample_yaws[i]);
    testViewsFile_output << data;
    testViewsFile_output << endl;
  }

  testViewsFile_output.close();

}

// Eigen::MatrixXf FastExplorationFSM::readMetric_Path(const std::string& filename) {
//     std::ifstream file(filename); // 打开文件
//     Eigen::MatrixXf matrix; // 定义Eigen矩阵

//     if (file.is_open()) {
//         std::vector<float> values;
//         std::string line;

//         while (std::getline(file, line)) {
//             // Check if the line contains "planned path"
//             if (line.find("planned path") != std::string::npos) {
//                 continue;  // Skip this line
//             }

//             std::istringstream iss(line);
//             std::string valueStr;

//             while (std::getline(iss, valueStr, ',')) { // 使用逗号作为分隔符解析数据
//                 float value = std::stof(valueStr); // 将解析的字符串转换为浮点数
//                 values.push_back(value);
//             }
//         }

//         cout << "values size = " << values.size() << endl;

//         // 获取数据的行数和列数
//         int cols = 4;
//         int rows = values.size() / cols;

//         // 将数据填充到Eigen矩阵
//         matrix.resize(rows, cols);
//         cout << "matrix size = " << matrix.rows() << " " << matrix.cols() << endl;

//         for (int i = 0; i < rows; ++i) {
//             for (int j = 0; j < cols; ++j) {
//                 matrix(i, j) = values[i * cols + j]; // 从索引0开始读取数据
//             }
//         }

//         file.close(); // 关闭文件
//     } else {
//         std::cout << "Failed to open the file.\n";
//     }
//     return matrix;
// }



std::vector<Eigen::MatrixXf> FastExplorationFSM::readMetric(const std::string& filename, int col_num) {
    std::ifstream file(filename); // Open the file
    std::vector<Eigen::MatrixXf> pathMatrices; // Vector to hold path matrices

    if (file.is_open()) {
        std::vector<float> values;
        std::string line;

        while (std::getline(file, line)) {
            if (line.find("planned num") != std::string::npos) {
                if (!values.empty()) {
                    int cols = col_num;
                    int rows = values.size() / cols;

                    Eigen::MatrixXf matrix(rows, cols);

                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            matrix(i, j) = values[i * cols + j];
                        }
                    }

                    pathMatrices.push_back(matrix);
                    values.clear();
                }
            } else {
                std::istringstream iss(line);
                std::string valueStr;

                while (std::getline(iss, valueStr, ',')) {
                    float value = std::stof(valueStr);
                    values.push_back(value);
                }
            }
        }

        // Process the last path
        if (!values.empty()) {
            int cols = col_num;
            int rows = values.size() / cols;

            Eigen::MatrixXf matrix(rows, cols);

            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    matrix(i, j) = values[i * cols + j];
                }
            }

            pathMatrices.push_back(matrix);
            values.clear();
        }

        file.close(); // Close the file
    } else {
        std::cout << "Failed to open the file.\n";
    }

    return pathMatrices;
}




}  // namespace fast_planner
