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

  // all_timer_ = nh.createTimer(ros::Duration(5.0), &FastExplorationManager::all_test,this);

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


// EVPP NBV
plannerResult FastExplorationManager::planExploreMotion_REC(
    const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, const Vector3d&  yaw) {

    ROS_WARN("[fast_exploration_evpp]come into EVPP NBV");
    auto cur_pos = pos;
    auto cur_pitch = yaw[1];
    auto cur_yaw = yaw[0];
    // Sampling views
    vector<Vector3d>  sampled_locations, sample_pos;
    vector<double>  sample_pitchs, sample_yaws,uncers;
    Eigen::Vector3d center(cur_pos); // 球心坐标
    double radius = 3.0; // 球的半径
    int num_samples = 100; // 采样点的数量
    ros::Time ts = ros::Time::now();
    sampled_locations = sample_location(center, num_samples, radius);
    vector<vector<double>> sample_views = sample_direction(sampled_locations);
    cout << "sample_time = " << (ros::Time::now() - ts).toSec() << endl;
    for (int i = 0; i < sample_views.size(); i++){
        sample_pos.push_back(Vector3d(sample_views[i][0],sample_views[i][1],sample_views[i][2]));
        sample_pitchs.push_back(sample_views[i][3]);
        sample_yaws.push_back(sample_views[i][4]);
    }
    // compute uncertainty 
    ros::Time start_time = ros::Time::now();
    uncers = get_uncertainty(sample_pos,sample_pitchs,sample_yaws);
    cout << "get_uncertainty time = " << (ros::Time::now() - start_time).toSec() << endl;
    std::vector<double> max_uncers; // 存储每三个元素中的最大值
    vector<Vector3d> max_sample_pos;
    vector<double> max_sample_pitchs, max_sample_yaws;
    for (int i = 0; i < uncers.size(); i += 3) {
        // 找到每三个元素中的最大值及其索引
        double max_val = uncers[i];
        int max_index = i;
        for (int j = i + 1; j < i + 3; j++) {
            if (uncers[j] > max_val) {
                max_val = uncers[j];
                max_index = j;
            }
        }
        max_uncers.push_back(max_val);
        // 将对应的sample_pos、sample_pitchs和sample_yaws加入到max_sample_pos、max_sample_pitchs和max_sample_yaws中
        max_sample_pos.push_back(sample_pos[max_index]);
        max_sample_pitchs.push_back(sample_pitchs[max_index]);
        max_sample_yaws.push_back(sample_yaws[max_index]);
    }

    surface_finder_->drawSampleViews(max_sample_pos,max_sample_pitchs,max_sample_yaws,max_uncers);
    ts = ros::Time::now();
    // 找到 max_uncers 向量的最大值
    int MAX_index = std::distance(max_uncers.begin(), std::max_element(max_uncers.begin(), max_uncers.end()));
    double MAX_uncer = max_uncers[MAX_index];
    Vector3d MAX_sample_pos = max_sample_pos[MAX_index];
    double MAX_sample_pitch = max_sample_pitchs[MAX_index];
    double MAX_sample_yaw = max_sample_yaws[MAX_index];
    // 对 uncers 向量进行除以最大值归一化
    for (size_t i = 0; i < max_uncers.size(); ++i) {
        max_uncers[i] /= MAX_uncer;
    }
    // train mlp
    train_mlp(max_sample_pos,max_uncers);
    cout << "train_mlp time = " << (ros::Time::now() - ts).toSec() << endl;
    mlp.eval();

    // IPP planning
    plannerResult result;
    vector<Vector3d> ipp_path;
    ViewNode::astar_->reset();
    ViewNode::astar_->setResolution(0.2);
    ViewNode::astar_->max_search_time_ = 0.5;
    ts = ros::Time::now();
    // 使用 std::bind 绑定成员函数和类实例
    auto bound_get_gain = std::bind(&FastExplorationManager::get_gain, this, std::placeholders::_1);
    // 传递绑定后的成员函数给 calculate 函数
    if (ViewNode::astar_->search_evpp(cur_pos, MAX_sample_pos, bound_get_gain) == Astar::REACH_END) {
      ipp_path = ViewNode::astar_->getPath();
      cout << "REACH_END"  << endl;
      result.res_state_ = SUCCEED;
      cout << "path =  " ;
      for (int i = 0; i < ipp_path.size(); i++){
        cout << ipp_path[i].transpose() << "->";
      }
      cout << endl;
    }
    else{
          cout << "FAIL"  << endl;
          result.res_state_ = FAIL;
          return result;
    }
    cout << "search_evpp time = " << (ros::Time::now() - ts).toSec() << endl;

    // select path views
    ed_->surf_path.clear();
    Vector3d next_pos;
    double next_pitch, next_yaw;
    int sec_num = ipp_path.size()-1;
    double yaw_err = MAX_sample_yaw-cur_yaw;
    if(yaw_err > M_PI) yaw_err -= 2*M_PI;
    if(yaw_err < -M_PI) yaw_err += 2*M_PI;
    for(int k=0; k<ipp_path.size()-1;k=k+1){
        next_pos = ipp_path[k+1];
        next_pitch = cur_pitch + (MAX_sample_pitch-cur_pitch)/sec_num*(k+1);
        next_yaw = cur_yaw + yaw_err/sec_num*(k+1);
        frontier_finder_->wrapYaw(next_yaw);

        result.res_pts_.push_back(next_pos);
        result.res_pitchs_.push_back(next_pitch);
        result.res_yaws_.push_back(next_yaw);
      
        ed_->surf_path.push_back(next_pos);
        Eigen::Vector4d path_point(next_pos[0],next_pos[1],next_pos[2],0.0);
        ed_->all_path.push_back(path_point);
    
    }

   
    
    return result;

  }



// process View and Path
plannerResult FastExplorationManager::processViewPath_REC(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws){

      plannerResult result;
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

vector<double> FastExplorationManager::convertJsonToVector(const Json::Value& jsonData) {
    std::vector<double> result;

    // 检查jsonData是否为数组类型
    if (jsonData.isArray()) {
        // 遍历数组元素
        for (const auto& value : jsonData) {
            // 检查数组元素是否为数值类型
            if (value.isNumeric()) {
                // 将数值类型的元素添加到结果向量中
                result.push_back(value.asDouble());
            }
        }
    }

    return result;
}

vector<double> FastExplorationManager::get_uncertainty(const vector<Eigen::Vector3d>& locations, const vector<double>& us, const vector<double>& vs){
	// cout << "get_uncertainty" << endl;
	// 创建Json::Value对象并填充数据
	Json::Value jsonData;
	for(int i=0; i < locations.size(); i++){
		Json::Value jsonView;
		jsonView["x"] = locations[i][0];
		jsonView["y"] = locations[i][1];
		jsonView["z"] = locations[i][2];
		jsonView["u"] = us[i];
		jsonView["v"] = vs[i];
		jsonData.append(jsonView);
	}

	// 将Json数据转换为字符串
    Json::StreamWriterBuilder writer;
    std::string jsonString = Json::writeString(writer, jsonData);

   // 发送POST请求到Django服务器
    std::chrono::seconds timeout(10);
    httplib::Client client("10.15.198.53", 7000);
    client.set_connection_timeout(timeout);
    client.set_read_timeout(timeout);
    auto response = client.Post("/get_uncertainty/", jsonString, "application/json");
 
     vector<double> uncertaintys;
    // 检查请求是否成功
    if (response && response->status == 200) {
        // std::cout << "请求成功：" << response->body << std::endl;
		// 解析响应体为JSON
        Json::Value jsonData;
        Json::CharReaderBuilder reader;
        std::string parseErrors;
         // 创建输入流并将响应体写入输入流
        std::istringstream is(response->body);
        bool parsingSuccessful = Json::parseFromStream(reader, is, &jsonData, &parseErrors);
		// 检查解析是否成功
        if (parsingSuccessful) {
            // 处理JSON数据
            // std::cout << "解析成功：" << jsonData.toStyledString() << std::endl;
			// cout << jsonData["uncers"] << endl;
			uncertaintys = convertJsonToVector(jsonData["uncers"]);
			// cout << result.size() << endl;
            
        } else {
            std::cout << "解析失败：" << parseErrors << std::endl;
		}


    } else {
        std::cout << "请求失败" << std::endl;
    }

    
    return uncertaintys;

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
  
    ROS_WARN("[fast_exploration_evpp]come into all_test");
    static int delay = 0;
    if (++delay < 2) return;

    vector<Vector3d>  sampled_locations, sample_pos;
    vector<double>  sample_pitchs, sample_yaws,uncers;
    Eigen::Vector3d center(0.0, 1.5, -2.0); // 球心坐标
    double radius = 3.0; // 球的半径
    int num_samples = 100; // 采样点的数量
    ros::Time ts = ros::Time::now();
    sampled_locations = sample_location(center, num_samples, radius);
    vector<vector<double>> sample_views = sample_direction(sampled_locations);
    cout << "sample_time = " << (ros::Time::now() - ts).toSec() << endl;
    for (int i = 0; i < sample_views.size(); i++){
        sample_pos.push_back(Vector3d(sample_views[i][0],sample_views[i][1],sample_views[i][2]));
        sample_pitchs.push_back(sample_views[i][3]);
        sample_yaws.push_back(sample_views[i][4]);
    }
    ros::Time start_time = ros::Time::now();
    uncers = get_uncertainty(sample_pos,sample_pitchs,sample_yaws);
    cout << "get_uncertainty time = " << (ros::Time::now() - start_time).toSec() << endl;
    std::vector<double> max_uncers; // 存储每三个元素中的最大值
    vector<Vector3d> max_sample_pos;
    vector<double> max_sample_pitchs, max_sample_yaws;
    for (int i = 0; i < uncers.size(); i += 3) {
        // 找到每三个元素中的最大值及其索引
        double max_val = uncers[i];
        int max_index = i;
        for (int j = i + 1; j < i + 3; j++) {
            if (uncers[j] > max_val) {
                max_val = uncers[j];
                max_index = j;
            }
        }
        max_uncers.push_back(max_val);
        // 将对应的sample_pos、sample_pitchs和sample_yaws加入到max_sample_pos、max_sample_pitchs和max_sample_yaws中
        max_sample_pos.push_back(sample_pos[max_index]);
        max_sample_pitchs.push_back(sample_pitchs[max_index]);
        max_sample_yaws.push_back(sample_yaws[max_index]);
    }

    surface_finder_->drawSampleViews(max_sample_pos,max_sample_pitchs,max_sample_yaws,max_uncers);
    ts = ros::Time::now();
    // 找到 uncers 向量的最大值
    int MAX_index = std::distance(max_uncers.begin(), std::max_element(max_uncers.begin(), max_uncers.end()));
    double MAX_uncer = max_uncers[MAX_index];
    Vector3d MAX_sample_pos = max_sample_pos[MAX_index];
    double MAX_sample_pitch = max_sample_pitchs[MAX_index];
    double MAX_sample_yaw = max_sample_yaws[MAX_index];
    // 对 uncers 向量进行除以最大值归一化
    for (size_t i = 0; i < max_uncers.size(); ++i) {
        max_uncers[i] /= MAX_uncer;
    }
    train_mlp(max_sample_pos,max_uncers);
    cout << "train_mlp time = " << (ros::Time::now() - ts).toSec() << endl;
    mlp.eval();

    Vector3d p1 = Vector3d(0.0, 1.5, -2.0);
    Vector3d p2 = MAX_sample_pos;
    vector<Vector3d> path;
    ViewNode::astar_->reset();
    ViewNode::astar_->setResolution(0.2);
    ViewNode::astar_->max_search_time_ = 0.5;

    ts = ros::Time::now();
    // 使用 std::bind 绑定成员函数和类实例
    auto bound_get_gain = std::bind(&FastExplorationManager::get_gain, this, std::placeholders::_1);
    // 传递绑定后的成员函数给 calculate 函数
    if (ViewNode::astar_->search_evpp(p1, p2, bound_get_gain) == Astar::REACH_END) {
      path = ViewNode::astar_->getPath();
      cout << "REACH_END"  << endl;
      cout << "path =  " ;
      for (int i = 0; i < path.size(); i++){
        cout << path[i].transpose() << "->";
      }
      cout << endl;
    }
    cout << "search_evpp time = " << (ros::Time::now() - ts).toSec() << endl;


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


Eigen::MatrixXf FastExplorationManager::readTextFile(const std::string& filename) {
    std::ifstream file(filename); // 打开文件
    Eigen::MatrixXf matrix; // 定义Eigen矩阵

    if (file.is_open()) {
        std::vector<float> values;
        std::string line;

        while (std::getline(file, line)) {
            std::istringstream iss(line);
            float value;

            while (iss >> value) {
                values.push_back(value);
            }
        }

        cout << "values size = " << values.size() << endl;

        // 获取数据的行数和列数
        int cols = 5;
        int rows = values.size() / cols;

        // 将数据填充到Eigen矩阵
        matrix.resize(rows, cols);
        cout << "matrix size = " << matrix.rows() << " " << matrix.cols() << endl;

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = values[i * cols + j]; // 从索引0开始读取数据
            }
        }

        file.close(); // 关闭文件
    } else {
        std::cout << "Failed to open the file.\n";
    }
// 
    return matrix;
}

vector<Eigen::Vector3d> FastExplorationManager::sample_location(Eigen::Vector3d node, int N, double R) {
    std::vector<Eigen::Vector3d> sampled_locations;
    int num_accepted_samples = 0;

    while (num_accepted_samples < N) {
        // 生成球坐标采样点
        double r = R * std::pow(rand_uniform_0_1(), 1.0 / 3.0);
        double theta = rand_uniform_neg_pi_to_pi() / 2.0;
        double phi = rand_uniform_neg_pi_to_pi();

        // 转换为三维坐标系中的点
        double x = r * std::cos(theta) * std::cos(phi);
        double y = r * std::cos(theta) * std::sin(phi);
        double z = r * std::sin(theta);
        Eigen::Vector3d sampled_location(x, y, z);
        sampled_location = sampled_location + node;

        if(sampled_location[1] > 2.5 || sampled_location[1]<0 ) continue;  // by zj
        // Qualified viewpoint is in bounding box and in safe region
        if (!frontier_finder_->edt_env_->sdf_map_->isInBox(sampled_location) ||
            frontier_finder_->edt_env_->sdf_map_->getInflateOccupancy(sampled_location) == 1|| frontier_finder_->isNearUnknown(sampled_location))
          continue;
        
        double dist = frontier_finder_->edt_env_->sdf_map_->getDistance(sampled_location);
        // cout << "dist = " << dist << endl;
        if (dist < 0.5) continue;  

        sampled_locations.push_back(sampled_location);
        num_accepted_samples++;




    }
    return sampled_locations;
}

vector<std::vector<double>> FastExplorationManager::sample_direction(const vector<Eigen::Vector3d>& locations) {
     std::vector<std::vector<double>> views;
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, 59); 

    for (const auto& location : locations) {
        double x = location[0];
        double y = location[1];
        double z = location[2];

        double dy = 1.7 - y;
        double u_center = std::atan2(-dy, 0.6);

        for (int i = 0; i < 5; i++) {
            double u = u_center - 30 / 180.0 * M_PI + i * 15 / 180.0 * M_PI;
            for (int j = 0; j < 12; j++) {
                double v = -M_PI + j * 2 * M_PI / 12.0;
                // Limit u to the range [-π/2 + π/9, π/2 - π/9]
                if (u < -M_PI / 2.0 + M_PI / 9.0) {
                    u = -M_PI / 2.0 + M_PI / 9.0;
                } else if (u > M_PI / 2.0 - M_PI / 9.0) {
                    u = M_PI / 2.0 - M_PI / 9.0;
                }

                views.push_back({x, y, z, u, v});
            }
        }
    }

    // Randomly select 3 viewpoints for each sampling point
    vector<vector<double>> sampled_views;
    for (size_t i = 0; i < views.size(); i += 60) {
        for (int j = 0; j < 3; j++) {
            int random_idx = dis(gen);
            sampled_views.push_back(views[i + random_idx]);
        }
    }

    return sampled_views;
}

void FastExplorationManager::train_mlp(vector<Eigen::Vector3d>& sample_pos, vector<double>& gains, int num_epochs, float learning_rate) {
    int num_samples = sample_pos.size(); // Number of data samples
    int num_features = 3; // Number of data features (Vector3d has 3 elements)

    // Convert sample_pos and uncers to torch tensors on the specified CUDA device
    torch::TensorOptions options(torch::kCUDA);
    options = options.device(torch::kCUDA, cuda_device); // Set the CUDA device
    torch::Tensor sample_pos_tensor = torch::empty({num_samples, num_features}, options);
    torch::Tensor gains_tensor = torch::empty({num_samples, 1}, options);

    // 在 for 循环中逐个赋值
    for (int i = 0; i < num_samples; ++i) {
        sample_pos_tensor[i][0] = sample_pos[i](0); // x
        sample_pos_tensor[i][1] = sample_pos[i](1); // y
        sample_pos_tensor[i][2] = sample_pos[i](2); // z
        gains_tensor[i][0] = gains[i]; // uncertainty
    }


    // Define the loss function and optimizer
    torch::nn::MSELoss loss_func;
    torch::optim::Adam optimizer(mlp.parameters(), learning_rate);

    // Randomly initialize model parameters
    mlp.randomInitializeModelParameters();

    // Train the network
    mlp.train();
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        torch::Tensor predictions = mlp.forward(sample_pos_tensor);
        torch::Tensor loss = loss_func(predictions, gains_tensor);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

}


double FastExplorationManager::get_gain(const Eigen::Vector3d& input) {
    // 将Eigen::Vector3d转换为torch::Tensor
    torch::Tensor input_tensor = torch::tensor({{input[0], input[1], input[2]}});
    input_tensor = input_tensor.to(device);

    // 进行MLP的前向传播，获取输出值
    torch::Tensor output = mlp.forward(input_tensor);

    // 将输出从 GPU 移动到 CPU，并将数据转换为 double 类型
    output = output.to(torch::kCPU);
    double result = output.item<double>();

    return result;
}



}  // namespace fast_planner
