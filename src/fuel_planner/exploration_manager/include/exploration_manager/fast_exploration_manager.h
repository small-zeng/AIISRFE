#ifndef _EXPLORATION_MANAGER_H_
#define _EXPLORATION_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <jsoncpp/json/json.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <torch/torch.h>
#include <cmath>
#include <random>

using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;




namespace fast_planner {
class EDTEnvironment;
class SDFMap;
class FastPlannerManager;
class FrontierFinder;
class SurfaceFinder;
class MLP;
struct ExplorationParam;
struct ExplorationData;
struct plannerResult;

class MLP : public torch::nn::Module {
public:
    MLP() :
        layer1(register_module("layer1", torch::nn::Linear(3, 64))),
        layer2_1(register_module("layer2_1", torch::nn::Linear(64, 64))),
        layer2_2(register_module("layer2_2", torch::nn::Linear(64, 64))),
        layer2_3(register_module("layer2_3", torch::nn::Linear(64, 64))),
        layer2_4(register_module("layer2_4", torch::nn::Linear(64, 64))),
        layer2_5(register_module("layer2_5", torch::nn::Linear(64, 64))),
        layer2_6(register_module("layer2_6", torch::nn::Linear(64, 64))),
        layer2_7(register_module("layer2_7", torch::nn::Linear(64, 64))),
        layer2_8(register_module("layer2_8", torch::nn::Linear(64, 64))),
        layer3(register_module("layer3", torch::nn::Linear(64, 1))) {
    }

   void randomInitializeModelParameters() {
        // 随机初始化网络的参数
        torch::NoGradGuard no_grad;  // 使用 no_grad 来避免计算梯度
        for (auto& parameter : parameters()) {
            if (parameter.dim() >= 2) {
                // 对权重参数使用 Kaiming 初始化
                torch::nn::init::xavier_uniform_(parameter);
            } else if (parameter.dim() == 1) {
                // 对偏置参数初始化为0
                // parameter.zero_();
                torch::nn::init::constant_(parameter, 0.5); // 这里将偏置参数初始化为0.5
            }
            // 其他情况下，参数可能是标量，跳过不做处理
        }
    }


    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x));
        x = torch::relu(layer2_1->forward(x));
        x = torch::relu(layer2_2->forward(x));
        x = torch::relu(layer2_3->forward(x));
        x = torch::relu(layer2_4->forward(x));
        x = torch::relu(layer2_5->forward(x));
        x = torch::relu(layer2_6->forward(x));
        x = torch::relu(layer2_7->forward(x));
        x = torch::relu(layer2_8->forward(x));
        x = layer3->forward(x);
        return x;
    }

private:
    torch::nn::Linear layer1, layer2_1, layer2_2, layer2_3, layer2_4, layer2_5, layer2_6, layer2_7, layer2_8, layer3;
};

enum EXPL_RESULT { NO_FRONTIER, FAIL, SUCCEED };

class FastExplorationManager {
public:
  FastExplorationManager();
  ~FastExplorationManager();

  void initialize(ros::NodeHandle& nh);

  plannerResult planExploreMotion_EXP(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc,
                        const Vector3d& yaw);
  plannerResult planExploreMotion_REC(const Vector3d& pos, const Vector3d& vel, const Vector3d& acc, 
                        const Vector3d&  yaw);

  // Benchmark method, classic frontier and rapid frontier
  int classicFrontier(const Vector3d& pos, const double& yaw);
  int rapidFrontier(const Vector3d& pos, const Vector3d& vel, const double& yaw, bool& classic);

  vector<double> convertJsonToVector(const Json::Value& jsonData);
  vector<double> get_uncertainty(const vector<Eigen::Vector3d>& locations, const vector<double>& us, const vector<double>& vs);
  void send_NBV(Eigen::Vector3d location, double  u, double  v);
  void drawTraj();
  void surf_test(const ros::TimerEvent& e);
  void all_test(const ros::TimerEvent& e);
  void drawAllTraj(const vector<Eigen::Vector4d>& plan_path);
  void findGlobalTour_All(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch, 
                  const vector<Vector3d>& next_poses, const vector<double>& next_pitchs, const vector<double>& next_yaws,vector<int>& indices);      
  void getFullCostMatrix_All(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch,
                      const vector<Vector3d>& next_poses, const vector<double>& next_pitchs, const vector<double>& next_yaws, 
                      Eigen::MatrixXd& mat);


  shared_ptr<ExplorationData> ed_;
  shared_ptr<ExplorationParam> ep_;
  shared_ptr<FastPlannerManager> planner_manager_;
  shared_ptr<FrontierFinder> frontier_finder_;
  shared_ptr<SurfaceFinder> surface_finder_;

  // frontier_call_flag
  bool frontier_call_flag;

  ros::Timer surf_timer_, all_timer_;
  
  // EVPP NBV 相关
  MLP mlp;
  int cuda_device;
  torch::Device device;
  // torch::TensorOptions options;
  Eigen::MatrixXf readTextFile(const std::string& filename); // for test
  vector<Eigen::Vector3d> sample_location(Eigen::Vector3d node, int N, double R);
  vector<std::vector<double>> sample_direction(const vector<Eigen::Vector3d>& locations);
  void train_mlp(vector<Eigen::Vector3d>& sample_pos, vector<double>& gains, int num_epochs = 300, float learning_rate = 0.001);
  double get_gain(const Eigen::Vector3d& input);
  // 生成 [0, 1) 之间的均匀分布随机数
  double rand_uniform_0_1() {
      static std::random_device rd;
      static std::mt19937 gen(rd());
      static std::uniform_real_distribution<> dis(0, 1);
      return dis(gen);
  }

  // 生成 [-π, π) 之间的均匀分布随机数
  double rand_uniform_neg_pi_to_pi() {
      return rand_uniform_0_1() * 2 * M_PI - M_PI;
  }

 

private:
  shared_ptr<EDTEnvironment> edt_environment_;
  shared_ptr<SDFMap> sdf_map_;



  // Find optimal tour for coarse viewpoints of all frontiers
  void findGlobalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d cur_yaw,
                      vector<int>& indices);

  // Refine local tour for next few frontiers, using more diverse viewpoints
  void refineLocalTour(const Vector3d& cur_pos, const Vector3d& cur_vel, const Vector3d& cur_yaw,
    const vector<vector<Vector3d>>& n_points, const vector<vector<double>>& n_pitchs,const vector<vector<double>>& n_yaws,
    vector<Vector3d>& refined_pts, vector<double>& refined_pitchs, vector<double>& refined_yaws) ;

  void shortenPath(vector<Vector3d>& path);
  plannerResult processViewPath_EXP(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws);
  plannerResult processViewPath_REC(const Vector3d & pos, const double & pitch, const double & yaw, 
      const vector<Vector3d> & next_poses, const vector<double> & next_pitchs, const vector<double> & next_yaws);

  void sampleViewInSphere(const Eigen::Vector3d& center, double& radius, 
                             Eigen::Vector3d& pos,  double& pitch,  double& yaw);
  void sampleViews(const Vector3d& cur_pos, vector<Vector3d>& sample_pos, vector<double>& sample_pitchs, 
                            vector<double>& sample_yaws);
  



public:
  typedef shared_ptr<FastExplorationManager> Ptr;
};

}  // namespace fast_planner

#endif