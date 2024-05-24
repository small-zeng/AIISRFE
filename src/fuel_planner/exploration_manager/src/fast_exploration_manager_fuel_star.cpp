// #include <fstream>
#include <exploration_manager/fast_exploration_manager.h>
#include <thread>
#include <iostream>
#include <fstream>
#include <lkh_tsp_solver/lkh_interface.h>
#include <active_perception/graph_node.h>
#include <active_perception/graph_search.h>
#include <active_perception/perception_utils.h>
#include <plan_env/raycast.h>
#include <plan_env/sdf_map.h>
#include <plan_env/edt_environment.h>
#include <active_perception/frontier_finder.h>
#include <active_perception/surface_finder.h>
#include <plan_manage/planner_manager.h>

#include <exploration_manager/expl_data.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <httplib.h>
#include <jsoncpp/json/json.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <chrono>
#include <torch/torch.h>
#include <cstdlib>
#include <cmath>
#include <algorithm> 

using namespace Eigen;

namespace fast_planner {
// SECTION interfaces for setup and query

FastExplorationManager::FastExplorationManager(): device(torch::kCUDA, 0){
  // 设置 CUDA_VISIBLE_DEVICES 环境变量为 "0,1,2,3"
  std::string devices = "0,1,2,3";
  std::string env_var_name = "CUDA_VISIBLE_DEVICES";
  std::string env_var_value = env_var_name + "=" + devices;
  int put_env_result = putenv(const_cast<char*>(env_var_value.c_str()));
  cuda_device = 3;
  device = torch::Device(torch::kCUDA, cuda_device);
  // options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  mlp.to(device);
}

FastExplorationManager::~FastExplorationManager() {
  ViewNode::astar_.reset();
  ViewNode::caster_.reset();
  ViewNode::map_.reset();
}

void FastExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new FastPlannerManager);
  planner_manager_->initPlanModules(nh);
  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;
  frontier_finder_.reset(new FrontierFinder(edt_environment_, nh));
  surface_finder_.reset(new SurfaceFinder(edt_environment_, nh));
  // view_finder_.reset(new ViewFinder(edt_environment_, nh));

  ed_.reset(new ExplorationData);
  ep_.reset(new ExplorationParam);

  nh.param("exploration/refine_local", ep_->refine_local_, true);
  nh.param("exploration/refined_num", ep_->refined_num_, -1);
  nh.param("exploration/refined_radius", ep_->refined_radius_, -1.0);
  nh.param("exploration/top_view_num", ep_->top_view_num_, -1);
  nh.param("exploration/max_decay", ep_->max_decay_, -1.0);
  nh.param("exploration/tsp_dir", ep_->tsp_dir_, string("null"));
  nh.param("exploration/relax_time", ep_->relax_time_, 1.0);

  nh.param("exploration/vm", ViewNode::vm_, -1.0);
  nh.param("exploration/am", ViewNode::am_, -1.0);
  nh.param("exploration/yd", ViewNode::yd_, -1.0);
  nh.param("exploration/ydd", ViewNode::ydd_, -1.0);
  nh.param("exploration/w_dir", ViewNode::w_dir_, -1.0);


  ViewNode::astar_.reset(new Astar);
  ViewNode::astar_->init(nh, edt_environment_);
  ViewNode::map_ = sdf_map_;
  

  double resolution_ = sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  sdf_map_->getRegion(origin, size);
  ViewNode::caster_.reset(new RayCaster);
  ViewNode::caster_->setParams(resolution_, origin);

  planner_manager_->path_finder_->lambda_heu_ = 1.0;
  // planner_manager_->path_finder_->max_search_time_ = 0.05;
  planner_manager_->path_finder_->max_search_time_ = 1.0;

  // Initialize TSP par file
  ofstream par_file(ep_->tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << ep_->tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << ep_->tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";

  // Analysis
  // ofstream fout;
  // fout.open("/home/boboyu/Desktop/RAL_Time/frontier.txt");
  // fout.close();
}

plannerResult FastExplorationManager::planExploreMotion_EXP(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d&  yaw) {
  frontier_call_flag = false;
  ros::Time t1 = ros::Time::now();
  auto t2 = t1;
  ed_->views_.clear();
  ed_->global_tour_.clear();
  plannerResult result;
  
  // std::cout << "\nstart pos: " << pos.transpose() << ", vel: " << vel.transpose()
  //           << ", acc: " << acc.transpose() << std::endl;

  // Search frontiers and group them into clusters
  frontier_finder_->searchFrontiers();

  double frontier_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Find viewpoints (x,y,z,yaw) for all frontier clusters and get visible ones' info
  frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);

  if (ed_->frontiers_.empty()) {
    ROS_WARN("No coverable frontier.");
    result.res_state_ = NO_FRONTIER;
    return result;
  }
  frontier_finder_->getTopViewpointsInfo(pos, ed_->points_, ed_ ->pitchs_, ed_->yaws_, ed_->averages_);

  double view_time = (ros::Time::now() - t1).toSec();
  ROS_WARN(
      "Frontier: %d, t: %lf, viewpoint: %d, t: %lf", ed_->frontiers_.size(), frontier_time,
      ed_->points_.size(), view_time);

  // Do global and local tour planning and retrieve the next viewpoint
  vector<Vector3d> next_poses;
  vector<double> next_pitchs, next_yaws;

  cout << "ed_->points_.size() = " << ed_->points_.size() << endl;
  ros::Time T_start = ros::Time::now();
  if (ed_->points_.size() > 1) {
    // Find the global tour passing through all viewpoints
    // Create TSP and solve by LKH
    // Optimal tour is returned as indices of frontier
    vector<int> indices;
    findGlobalTour(pos, vel, yaw, indices);
    int knum = min(int(indices.size()), 10);
    
    // Choose the next viewpoint from global tour
    for(int k=0; k < knum; k++){
        next_poses.push_back(ed_->points_[indices[k]]);
        next_pitchs.push_back(ed_->pitchs_[indices[k]]);
        next_yaws.push_back(ed_->yaws_[indices[k]]);
      }
    
  } else if (ed_->points_.size() == 1) {
    // Only 1 destination, no need to find global tour through TSP
    ed_->global_tour_ = { pos, ed_->points_[0] };
    ed_->refined_tour_.clear();

    next_poses.push_back(ed_->points_[0]);
    next_pitchs.push_back(ed_->pitchs_[0]);
    next_yaws.push_back(ed_->yaws_[0]);
    
  } else
    ROS_ERROR("Empty destination.");
  ed_->T_atsp += (ros::Time::now()-T_start).toSec();
  
  double total = (ros::Time::now() - t2).toSec();
  ROS_WARN("Total time: %lf", total);
  ROS_ERROR_COND(total > 0.1, "Total time too long!!!");

  
  // if(next_poses.size()>2){
  //   double dis = (next_poses[0]-pos).norm();
  //   cout << "next_poses dis = " << dis << endl;
  //   if(dis < 0.2){
  //     next_poses.erase(next_poses.begin());
  //     next_pitchs.erase(next_pitchs.begin());
  //     next_yaws.erase(next_yaws.begin());
  //   }
  // }

  // print global views
  cout << "global views: ";
  for(int i = 0; i < next_poses.size(); i++){
    cout <<  next_poses[i].transpose() << ", ";
  }
  cout << endl;


  result = processViewPath_EXP(pos,0,0,next_poses,next_pitchs,next_yaws);

  return result;
}

// process View and Path
plannerResult FastExplorationManager::processViewPath_EXP(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws){

  // process view and path
  int NBV_NUM = 0,NBV_NUM_MAX = 30;
  int n = 2;
  Eigen::Vector3d next_pos(pos);
  double next_pitch = pitch, next_yaw = yaw;
  int N = min(int(next_poses.size()),n);
  // int knum = min(int(indices.size()), ep_->refined_num_);
  cout << "Send NBV num = " << N << endl;
  Eigen::Vector3d start_pos = pos;
  double start_pitch = pitch;
  double start_yaw = yaw;
  plannerResult result;


  ed_->exp_path.clear();  //清空路径
  for (int i = 0; i < N; ++i) {
    // Generate trajectory of x,y,z
    planner_manager_->path_finder_->reset();
    planner_manager_->path_finder_->max_search_time_ = 0.2;
    cout << "A star searching: " << start_pos.transpose() << " ----> " << next_poses[i].transpose() << endl;
    if (planner_manager_->path_finder_->search(start_pos, next_poses[i]) != Astar::REACH_END) {
      ROS_ERROR("[EXP] No path to next viewpoint");
      break;
    }
    ed_->path_next_goal_ = planner_manager_->path_finder_->getPath();

    // // log path
    // cout << "path = ";
    // ros::Time current_time = ros::Time::now();
    // for(int j=0; j < ed_->path_next_goal_.size(); j++){
    //     cout << ed_->path_next_goal_[j].transpose()<< "; ";
    //     }
    // cout << endl;
    
    //select viewpoint
    int sec_num = ed_->path_next_goal_.size()-1;
    double yaw_err = next_yaws[i]-start_yaw;
    if(yaw_err > M_PI) yaw_err -= 2*M_PI;
    if(yaw_err < -M_PI) yaw_err += 2*M_PI;
    for(int k=0; k<ed_->path_next_goal_.size()-1;k=k+1){
       if (NBV_NUM >=NBV_NUM_MAX) break;
        next_pos = ed_->path_next_goal_[k+1];
        next_pitch = start_pitch + (next_pitchs[i]-start_pitch)/sec_num*(k+1);
        next_yaw = start_yaw + yaw_err/sec_num*(k+1);
        frontier_finder_->wrapYaw(next_yaw);
        int m = 1;
        // if (ed_->is_only_exp == false)  m =2;
        if(k%m==0){
          result.res_pts_.push_back(next_pos);
          result.res_pitchs_.push_back(next_pitch);
          result.res_yaws_.push_back(next_yaw);
          NBV_NUM ++;
        }

        ed_->exp_path.push_back(next_pos);
        Eigen::Vector4d path_point(next_pos[0],next_pos[1],next_pos[2],1.0);
        ed_->all_path.push_back(path_point);
    

    }

    start_pos = next_poses[i];
    start_pitch = next_pitchs[i];
    start_yaw = next_yaws[i];

   
  }


  if(result.res_pts_.size() == 0){
    result.res_state_ = FAIL;
   
  }
  else{
    result.res_state_ = SUCCEED;
    
  }
  
  // frontier_call_flag = true;
  return result;




}

plannerResult FastExplorationManager::planExploreMotion_REC(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d&  yaw) {
  
  plannerResult result;
  return result;
}

// process View and Path
plannerResult FastExplorationManager::processViewPath_REC(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws){

  plannerResult result;
  return result;

}

void FastExplorationManager::shortenPath(vector<Vector3d>& path) {
  if (path.empty()) {
    ROS_ERROR("Empty path to shorten");
    return;
  }
  // Shorten the tour, only critical intermediate points are reserved.
  const double dist_thresh = 3.0;
  vector<Vector3d> short_tour = { path.front() };
  for (int i = 1; i < path.size() - 1; ++i) {
    if ((path[i] - short_tour.back()).norm() > dist_thresh)
      short_tour.push_back(path[i]);
    else {
      // Add waypoints to shorten path only to avoid collision
      ViewNode::caster_->input(short_tour.back(), path[i + 1]);
      Eigen::Vector3i idx;
      while (ViewNode::caster_->nextId(idx) && ros::ok()) {
        if (edt_environment_->sdf_map_->getInflateOccupancy(idx) == 1 ||
            edt_environment_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
          short_tour.push_back(path[i]);
          break;
        }
      }
    }
  }
  if ((path.back() - short_tour.back()).norm() > 1e-3) short_tour.push_back(path.back());

  // Ensure at least three points in the path
  if (short_tour.size() == 2)
    short_tour.insert(short_tour.begin() + 1, 0.5 * (short_tour[0] + short_tour[1]));
  path = short_tour;
}

void FastExplorationManager::findGlobalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    vector<int>& indices) {
  auto t1 = ros::Time::now();
  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  // frontier_finder_->updateFrontierCostMatrix();
  frontier_finder_->getFullCostMatrix(cur_pos, cur_vel, cur_yaw, cost_mat);
  const int dimension = cost_mat.rows();
  ROS_WARN("exploration dimension = %d", dimension);

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Write params and cost matrix to problem file
  ofstream prob_file(ep_->tsp_dir_ + "/single.tsp");
  // Problem specification part, follow the format of TSPLIB

  string prob_spec = "NAME : single\nTYPE : ATSP\nDIMENSION : " + to_string(dimension) +
      "\nEDGE_WEIGHT_TYPE : "
      "EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n";

  // string prob_spec = "NAME : single\nTYPE : TSP\nDIMENSION : " + to_string(dimension) +
  //     "\nEDGE_WEIGHT_TYPE : "
  //     "EXPLICIT\nEDGE_WEIGHT_FORMAT : LOWER_ROW\nEDGE_WEIGHT_SECTION\n";

  prob_file << prob_spec;
  // prob_file << "TYPE : TSP\n";
  // prob_file << "EDGE_WEIGHT_FORMAT : LOWER_ROW\n";
  // Problem data part
  const int scale = 100;
  // Use Asymmetric TSP
  for (int i = 0; i < dimension; ++i) {
    for (int j = 0; j < dimension; ++j) {
      int int_cost = cost_mat(i, j) * scale;
      prob_file << int_cost << " ";
    }
    prob_file << "\n";
  }
  prob_file << "EOF";
  prob_file.close();

  // Call LKH TSP solver
  solveTSPLKH((ep_->tsp_dir_ + "/single.par").c_str());

  cout << "test 1"<< endl;
  // Read optimal tour from the tour section of result file
  ifstream res_file(ep_->tsp_dir_ + "/single.txt");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  if (false) {
    // Read path for Symmetric TSP formulation
    getline(res_file, res);  // Skip current pose
    getline(res_file, res);
    int id = stoi(res);
    bool rev = (id == dimension);  // The next node is virutal depot?

    while (id != -1) {
      indices.push_back(id - 2);
      getline(res_file, res);
      id = stoi(res);
    }
    if (rev) reverse(indices.begin(), indices.end());
    indices.pop_back();  // Remove the depot

  } else {
    // Read path for ATSP formulation
    while (getline(res_file, res)) {
      // Read indices of frontiers in optimal tour
      int id = stoi(res);
      if (id == 1)  // Ignore the current state
        continue;
      if (id == -1) break;
      indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
    }
  }

  res_file.close();

  cout << "test 2"<< endl;

  // Get the path of optimal tour from path matrix
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->global_tour_);
  cout << "test 3"<< endl;

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}


void FastExplorationManager::send_NBV(Eigen::Vector3d location, double  u, double  v){
    
    // cout << "send NBV:  " << location.transpose() << ", " << u << ", " << v << endl;
    // httplib::Client cli("192.168.31.17", 7200);
    httplib::Client cli("10.190.72.35", 7200);
    httplib::Headers headers;
    httplib::Params params;
    double x = -location[0];
    double y = location[1];
    double z = location[2];
    u = u /M_PI*180;
    v = -v /M_PI*180;
    // cout << "xyzuv = " << x << ", " << y << ", " << z << ", " << u << ", " << v << endl; 
    params.insert({"x",to_string(x)});
    params.insert({"y",to_string(y)});
    params.insert({"z",to_string(z)});
    params.insert({"u",to_string(u)});
    params.insert({"v",to_string(v)});

    // cout << location << u << v << endl;
    if (auto res = cli.Get("/",params ,headers)){

      if (res->status == 200) {
        cout << "succeed  " << res->body << endl;
      }
    } 
    else {
      auto err = res.error();
      cout << "fail  " << err << endl;
    }


}


void FastExplorationManager::drawAllTraj(const vector<Eigen::Vector4d>& plan_path){
    // 创建路径消息的可视化对象
    visualization_msgs::Marker traj_marker;
    ros::Time current_time = ros::Time::now();
    traj_marker.header.frame_id = "world";
    traj_marker.header.stamp = current_time;
    traj_marker.ns = "trajectory";
    traj_marker.id = 0;
    traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
    traj_marker.action = 0u;
    traj_marker.pose.orientation.w = 1.0;
    traj_marker.scale.x = 0.02; // 设置线条的宽度


    // 设置路径线条的顶点
    for (int i = 0; i < plan_path.size(); i++) {
      geometry_msgs::Point point;
      point.x = plan_path[i][0];
      point.y = plan_path[i][1];
      point.z = plan_path[i][2];
      traj_marker.points.push_back(point);

      std_msgs::ColorRGBA color;
      // 设置路径线条的颜色
      if (plan_path[i][3] == 1){
          color.r = 0.0;
          color.g = 1.0;
          color.b = 0.0;
          color.a = 1.0; // 不透明度

      } else{
          color.r = 1.0;
          color.g = 0.0;
          color.b = 0.0;
          color.a = 1.0; // 不透明度
      }

      traj_marker.colors.push_back(color);
    }

    // 发布可视化消息
    ed_->all_traj_pub.publish(traj_marker);

    // // 循环输出plan_path，plan_path是全局路径
    // cout << "surface plan_path = " << endl;
    // for (int i = 0; i < plan_path.size(); i++) {
    //     cout  << plan_path[i].transpose() << "->";
    // }
    // cout << endl;

  }



}  // namespace fast_planner
