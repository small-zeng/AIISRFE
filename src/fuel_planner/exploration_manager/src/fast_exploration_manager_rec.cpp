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
#include <cmath>
#include <chrono>
// #include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
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

  // surf_timer_ = nh.createTimer(ros::Duration(2.0), &FastExplorationManager::surf_test,this);
  // all_timer_ = nh.createTimer(ros::Duration(2.0), &FastExplorationManager::all_test,this);

}

plannerResult FastExplorationManager::planExploreMotion_EXP(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d&  yaw) {
  
  plannerResult result;
  return result;
}

// process View and Path
plannerResult FastExplorationManager::processViewPath_EXP(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws){

  plannerResult result;
  return result;

}



plannerResult FastExplorationManager::planExploreMotion_REC(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d&  yaw) {
   
  // All the frontiers are covered, only do surface reconstruction tasks
  if(ed_->mode_num == 1){
    surface_finder_->points_radius_ = 4.0;
    surface_finder_->cluster_num_ = 20;
    surface_finder_->cluster_radius_ = 1.2;
    surface_finder_->max_path_length_ = 30.0;
  }

  auto cur_pos = pos;
  auto cur_pitch = yaw[1];
  auto cur_yaw = yaw[0];
  auto cur_vel = Eigen::Vector3d(0,0,0);
  auto cur_dir = Eigen::Vector3d(0,0,0);
  vector<Vector3d> next_poses;
  vector<double> next_pitchs, next_yaws;
  plannerResult result;

    
  ros::Time ts = ros::Time::now();
  // get get surface uncertainty
  int model_index;
  std::string model_dir;
  httplib::Client client(ed_->local_ip_, 7000);  // 根据你的实际情况修改主机和端口
  std::string body = "";  // 设置请求体为空字符串
  httplib::Result res = client.Post("/get_surface_uncertainty/", body, "application/json");
  // 检查请求是否成功
  if (res && res->status == 200) {
      std::cout << "请求成功：" << res->body << std::endl;
      // 返回为 re = {"model_index":epoch,"model_dir":path}，解析出model_index和model_dir
      Json::CharReaderBuilder readerBuilder;
      Json::Value root;
      std::string err;

      // 解析JSON字符串
      std::istringstream jsonStream(res->body);
      bool parsingSuccessful = Json::parseFromStream(readerBuilder, jsonStream, &root, &err);

      // 检查解析是否成功
      if (!parsingSuccessful) {
          std::cout << "解析JSON失败：" << err << std::endl;
          result.res_state_ = FAIL;
          return result;
      }

      // 提取model_index和model_dir的值
      model_index = root["model_index"].asInt();
      model_dir = root["model_dir"].asString();

      // 使用提取到的值进行后续操作
      std::cout << "model_index: " << model_index << std::endl;
      std::cout << "model_dir: " << model_dir << std::endl;
    

  } 
  else {
      std::cout << "请求失败：" << res->body << std::endl;
      result.res_state_ = FAIL;
      return result;
  }
  ROS_WARN("get_surface_uncertainty time = %f", (ros::Time::now() - ts).toSec());

  // reconstruction
  // get surface points from file
  surface_finder_->mesh_data_base_dir = model_dir;
  surface_finder_->get_surface_points(cur_pos);
  // 聚类
  surface_finder_->surface_clusters.clear();
  surface_finder_->clusterPoints(surface_finder_->points, surface_finder_->points_clusters);
  surface_finder_->computeSurfaceclustersToVisit();
  ed_->surf_path.clear();  //清空路径
  ed_->surf_points_.clear();
  ed_->surf_yaws_.clear();
  ed_->surf_pitchs_.clear();
  ros::Time T_start = ros::Time::now();
  if (surface_finder_->surface_clusters.size() > 1) {
      vector<int> indices;
      surface_finder_->findGlobalTour(cur_pos, cur_yaw, cur_pitch, indices); 
      // Get the path of optimal tour from path matrix
      cout << "indices.size() = " << indices.size() << endl;
      surface_finder_->getPathForTour(cur_pos, indices, ed_->surf_path,ed_->surf_points_,ed_->surf_yaws_,ed_->surf_pitchs_);
      cout << "indices.size() = " << indices.size() << endl;
      cout << "ed_->surf_points_.size() = " << ed_->surf_points_.size() << endl;
      if (ed_->surf_points_.size() > 0){
        int N = int(indices.size());
        if (ed_->mode_num == 2)  N = min(int(indices.size()), 5);
        for(int i = 0; i < N; i++){
          cout << "indices = " << indices[i] << endl;
          next_poses.push_back(ed_->surf_points_[i]);
          next_pitchs.push_back(ed_->surf_pitchs_[i]);
          next_yaws.push_back(ed_->surf_yaws_[i]);
        }
      }
    
  }else if (surface_finder_->surface_clusters.size() == 1) {
      // Only 1 destination, no need to find global tour through TSP
      next_poses.push_back(surface_finder_->surface_clusters[0].viewpoints_.front().pos_);
      next_pitchs.push_back(surface_finder_->surface_clusters[0].viewpoints_.front().pitch_);
      next_yaws.push_back(surface_finder_->surface_clusters[0].viewpoints_.front().yaw_);
  } else 
    ROS_ERROR("[REC] Empty destination.");
  ed_->T_atsp += (ros::Time::now()-T_start).toSec();
  
  cout << "[REC] next_poses = " ;
  for (int i = 0; i < next_poses.size(); ++i) {
    cout  << next_poses[i].transpose() << "-->";
  }
  cout << endl;
  surface_finder_->visualize(ed_->surf_path);

  result = processViewPath_REC(cur_pos,cur_pitch,cur_yaw, next_poses,next_pitchs,next_pitchs);

  return result;


  }



// process View and Path
plannerResult FastExplorationManager::processViewPath_REC(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws){

     // process view and path
    int SURF_NUM = 0,SURF_NUM_MAX = 20, n = 10;
    if (ed_->mode_num == 1) {SURF_NUM_MAX = 20; n = 20;}
    Eigen::Vector3d next_pos(pos);
    double next_pitch = pitch, next_yaw = yaw;
    int N = min(int(next_poses.size()),n);
    Eigen::Vector3d start_pos = pos;
    double start_pitch = pitch;
    double start_yaw = yaw;
    plannerResult result;

  ed_->surf_path.clear();  //清空路径
  for (int i = 0; i < N; ++i) {
    // Generate trajectory of x,y,z
    planner_manager_->path_finder_->reset();
    planner_manager_->path_finder_->max_search_time_ = 0.2;
    cout << "A star searching: " << start_pos.transpose() << " ----> " << next_poses[i].transpose() << endl;
    if (planner_manager_->path_finder_->search(start_pos, next_poses[i]) != Astar::REACH_END) {
      ROS_ERROR("[REC] No path to next viewpoint");
      break;
    }
    ed_->surf_path_next_goal_ = planner_manager_->path_finder_->getPath();
    
    //select viewpoint
    int sec_num = ed_->surf_path_next_goal_.size()-1;
    double yaw_err = next_yaws[i]-start_yaw;
    if(yaw_err > M_PI) yaw_err -= 2*M_PI;
    if(yaw_err < -M_PI) yaw_err += 2*M_PI;
    for(int k=0; k<ed_->surf_path_next_goal_.size()-1;k=k+1){
       if (SURF_NUM >=SURF_NUM_MAX) break;
        next_pos = ed_->surf_path_next_goal_[k+1];
        next_pitch = start_pitch + (next_pitchs[i]-start_pitch)/sec_num*(k+1);
        next_yaw = start_yaw + yaw_err/sec_num*(k+1);
        frontier_finder_->wrapYaw(next_yaw);
        int m = 2;
        if (ed_->mode_num == 1)  m = 4;
        if(k%m==0){
          result.res_pts_.push_back(next_pos);
          result.res_pitchs_.push_back(next_pitch);
          result.res_yaws_.push_back(next_yaw);
          SURF_NUM ++;
        }

        ed_->surf_path.push_back(next_pos);
        Eigen::Vector4d path_point(next_pos[0],next_pos[1],next_pos[2],0.0);
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
  
    
  // visualize
  surface_finder_->visualize(ed_->surf_path);
  // thread visualize_thread(&SurfaceFinder::visualize, surface_finder_, std::ref(ed_->surf_path));
  // visualize_thread.detach();
  drawAllTraj(ed_->all_path);



return result;

}



void FastExplorationManager::findGlobalTour(
    const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
    vector<int>& indices) {
  auto t1 = ros::Time::now();

  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  frontier_finder_->updateFrontierCostMatrix();
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

  // Get the path of optimal tour from path matrix
  frontier_finder_->getPathForTour(cur_pos, indices, ed_->global_tour_);

  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);
}


void FastExplorationManager::send_NBV(Eigen::Vector3d location, double  u, double  v){
    
    // cout << "send NBV:  " << location.transpose() << ", " << u << ", " << v << endl;
    httplib::Client cli("10.13.21.209", 7200);
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


void FastExplorationManager::all_test(const ros::TimerEvent& e){
  
  auto cur_pos = Eigen::Vector3d(0,1.5,-2);
  auto cur_pitch = 0.0;
  auto cur_yaw = 0.0;
  auto cur_vel = Eigen::Vector3d(0,0,0);
  auto cur_dir = Eigen::Vector3d(0,0,0);
  vector<Vector3d> next_poses;
  vector<double> next_pitchs, next_yaws;

  // exploration
  ros::Time ts = ros::Time::now();
  frontier_finder_->searchFrontiers();
  frontier_finder_->computeFrontiersToVisit();
  frontier_finder_->getFrontiers(ed_->frontiers_);
  frontier_finder_->getFrontierBoxes(ed_->frontier_boxes_);
  frontier_finder_->getDormantFrontiers(ed_->dead_frontiers_);

  frontier_finder_->getTopViewpointsInfo(cur_pos, ed_->points_, ed_ ->pitchs_, ed_->yaws_, ed_->averages_);

  ed_->global_tour_.clear();
  if (ed_->points_.size() > 1) {
    vector<int> indices;
    findGlobalTour(cur_pos, cur_vel, cur_dir, indices);
    int knum = min(int(indices.size()), 10);
    // Choose the next viewpoint from global tour
    for(int k=0; k < knum; k++){
        next_poses.push_back(ed_->points_[indices[k]]);
        next_pitchs.push_back(ed_->pitchs_[indices[k]]);
        next_yaws.push_back(ed_->yaws_[indices[k]]);
      }
  
  }else if (ed_->points_.size() == 1) {
    // Only 1 destination, no need to find global tour through TSP
    next_poses.push_back(ed_->points_[0]);
    next_pitchs.push_back(ed_->pitchs_[0]);
    next_yaws.push_back(ed_->yaws_[0]);
  } else 
    ROS_ERROR("[EXP] Empty destination.");
  cout << "[EXP] next_poses = " ;
  for (int i = 0; i < next_poses.size(); ++i) {
    cout  << next_poses[i].transpose() << "-->";
  }
  cout << endl;
  ROS_WARN("exploration time = %f", (ros::Time::now() - ts).toSec());
  frontier_finder_->drawFirstViews();
  frontier_finder_->drawTraj(ed_->global_tour_);


  // reconstruction
  ts = ros::Time::now();
  // get surface points from file
  surface_finder_->get_surface_points(cur_pos);
  // 聚类
  surface_finder_->surface_clusters.clear();
  surface_finder_->clusterPoints(surface_finder_->points, surface_finder_->points_clusters);
  surface_finder_->computeSurfaceclustersToVisit();
  vector<int> indices;
  ed_->surf_path.clear();  //清空路径
  ed_->surf_points_.clear();
  ed_->surf_yaws_.clear();
  ed_->surf_pitchs_.clear();
  if (surface_finder_->surface_clusters.size() > 1) {
      surface_finder_->findGlobalTour(cur_pos, cur_yaw, cur_pitch, indices); 
      // Get the path of optimal tour from path matrix
      surface_finder_->getPathForTour(cur_pos, indices, ed_->surf_path,ed_->surf_points_,ed_->surf_yaws_,ed_->surf_pitchs_);
      for(int i = 0; i < ed_->surf_points_.size(); i++){
        next_poses.push_back(ed_->surf_points_[i]);
        next_pitchs.push_back(ed_->surf_pitchs_[i]);
        next_yaws.push_back(ed_->surf_yaws_[i]);
      }
    
  }else if (surface_finder_->surface_clusters.size() == 1) {
      // Only 1 destination, no need to find global tour through TSP
      next_poses.push_back(surface_finder_->surface_clusters[0].viewpoints_.front().pos_);
      next_pitchs.push_back(surface_finder_->surface_clusters[0].viewpoints_.front().pitch_);
      next_yaws.push_back(surface_finder_->surface_clusters[0].viewpoints_.front().yaw_);
  } else 
    ROS_ERROR("[REC] Empty destination.");

  cout << "[REC] next_poses = " ;
  for (int i = 0; i < next_poses.size(); ++i) {
    cout  << next_poses[i].transpose() << "-->";
  }
  cout << endl;
  surface_finder_->visualize(ed_->surf_path);

  // All
  indices.clear();
  ed_->all_points_.clear();
  ed_->all_pitchs_.clear();
  ed_->all_yaws_.clear();
  if (next_poses.size() > 1) {
     findGlobalTour_All(cur_pos, cur_yaw, cur_pitch, next_poses, next_pitchs, next_yaws, indices);
     for(int i = 0; i < indices.size(); i++){
        ed_->all_points_.push_back(next_poses[indices[i]]);
        ed_->all_pitchs_.push_back(next_pitchs[indices[i]]);
        ed_->all_yaws_.push_back(next_yaws[indices[i]]);
      }

  }else if (next_poses.size() == 1) {
      // Only 1 destination, no need to find global tour through TSP
      ed_->all_points_.push_back(next_poses[0]);
      ed_->all_pitchs_.push_back(next_pitchs[0]);
      ed_->all_yaws_.push_back(next_yaws[0]);
  } else 
    ROS_ERROR("[All] Empty destination.");


  cout << "[All] next_poses = " ;
  for (int i = 0; i < ed_->all_points_.size(); ++i) {
    cout  << ed_->all_points_[i].transpose() << "-->";
  }
  cout << endl;
  processViewPath_REC(cur_pos,cur_pitch,cur_yaw, ed_->all_points_,ed_->all_pitchs_,ed_->all_yaws_);




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


void FastExplorationManager::findGlobalTour_All(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch,
                           const vector<Vector3d>& next_poses, const vector<double>& next_pitchs, const vector<double>& next_yaws, vector<int>& indices) {

  auto t1 = ros::Time::now();
  // Get cost matrix for current state and clusters
  Eigen::MatrixXd cost_mat;
  getFullCostMatrix_All(cur_pos, cur_yaw, cur_pitch, next_poses, next_pitchs, next_yaws, cost_mat);
  const int dimension = cost_mat.rows();
  ROS_WARN("all dimension = %d", dimension);

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


  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);


}

// Our TSP Matrix
/*
    0 V V V  
    0 V V V  
    0 V V V  
    0 V V V 
*/
void FastExplorationManager::getFullCostMatrix_All(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch,
                         const vector<Vector3d>& next_poses, const vector<double>& next_pitchs, const vector<double>& next_yaws,
                         Eigen::MatrixXd& mat) {

    // Use Asymmetric TSP
    int dimen = next_poses.size();
    mat.resize(dimen + 1, dimen + 1);
    // Fill block for next_poses
    for (int i = 1; i < next_poses.size() + 1; ++i) {
      for (int j = 1; j < next_poses.size() + 1; ++j) {
        vector<Vector3d> path;
        double cost_ij = ViewNode::computeCostNew(next_poses[i-1], next_poses[j-1], next_yaws[i-1],next_yaws[j-1], next_pitchs[i-1], next_pitchs[j-1], path);
        mat(i, j) = cost_ij;
      }
    }


    // Fill block from current state to clusters
    mat.leftCols<1>().setZero();
    for (int j = 1; j < next_poses.size() + 1; ++j) {
      vector<Vector3d> path;
      mat(0, j) = ViewNode::computeCostNew(cur_pos, next_poses[j-1], cur_yaw, next_yaws[j-1], cur_pitch, next_pitchs[j-1], path);
    }
  
}



}  // namespace fast_planner
