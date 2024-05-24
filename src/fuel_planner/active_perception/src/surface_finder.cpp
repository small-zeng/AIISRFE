#include <active_perception/surface_finder.h>
#include <plan_env/sdf_map.h>
#include <plan_env/raycast.h>
// #include <path_searching/astar2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <plan_env/edt_environment.h>
#include <active_perception/perception_utils.h>
#include <active_perception/graph_node.h>

#include <pcl/filters/voxel_grid.h>
#include <Eigen/Eigenvalues>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/opencv.hpp>
#include <lkh_tsp_solver/lkh_interface.h>

#include <pcl/io/ply_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <opencv2/opencv.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <random>


namespace fast_planner {
SurfaceFinder::SurfaceFinder(const EDTEnvironment::Ptr& edt, ros::NodeHandle& nh) {
  this->edt_env_ = edt;

  // nh.param<string>("surface/mesh_mkdir", mesh_base_dir, "");
  nh.param<string>("surface/mesh_data_mkdir", mesh_data_base_dir, "");
  nh.param<string>("surface/tsp_dir", tsp_dir_, "");
  nh.param("surface/candidate_dphi", candidate_dphi_, -1.0);
  nh.param("surface/candidate_rmax", candidate_rmax_, -1.0);
  nh.param("surface/candidate_rmin", candidate_rmin_, -1.0);
  nh.param("surface/candidate_rnum", candidate_rnum_, -1);
  nh.param("surface/min_candidate_clearance", min_candidate_clearance_, -1.0);
  nh.param("surface/min_visib_num", min_visib_num_, -1);
  nh.param("surface/down_sample", down_sample_, -1);
  nh.param("surface/cluster_num", cluster_num_, -1);
  nh.param("surface/cluster_radius", cluster_radius_, -1.0);
  nh.param("surface/points_radius", points_radius_, -1.0);
  nh.param("surface/max_path_length", max_path_length_, -1.0);
  

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);
  percep_utils_.reset(new PerceptionUtils(nh));

  mesh_pub = nh.advertise<sensor_msgs::PointCloud2>("/surface/mesh", 10);
  surface_clusters_pub = nh.advertise<sensor_msgs::PointCloud2>("/surface/surface_clusters", 10);
  views_vis_pub = nh.advertise<visualization_msgs::MarkerArray>("/surface/all_views", 10);
  // surface_traj_pub = nh.advertise<nav_msgs::Path>("/surface/plan_traj",1, true); 
  traj_pub = nh.advertise<visualization_msgs::Marker>("/surface/plan_traj", 10);


  surface_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
  surface_clusters_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();

  // auto cur_pos = Eigen::Vector3d(0,1.5,-2);
  // get_surface_points(cur_pos);
  // test_timer_ = nh.createTimer(ros::Duration(2.0), &SurfaceFinder::test,this);

// Initialize TSP par file
  ofstream par_file(tsp_dir_ + "/single.par");
  par_file << "PROBLEM_FILE = " << tsp_dir_ << "/single.tsp\n";
  par_file << "GAIN23 = NO\n";
  par_file << "OUTPUT_TOUR_FILE =" << tsp_dir_ << "/single.txt\n";
  par_file << "RUNS = 1\n";



}

SurfaceFinder::~SurfaceFinder() {
}

void SurfaceFinder::pubmeshCallback() {
  
  // send mesh cloud
  // 定义需要删除的点的条件，这里假设删除 y 坐标大于 1.6 的点
  // float minY = 1.6;
  // pcl::PointCloud<pcl::PointXYZRGB>::iterator it = surface_cloud->begin();
  // while (it != surface_cloud->end()) {
  //   if (it->y > minY) {
  //     it = surface_cloud->erase(it); // 删除满足条件的点
  //   } else {
  //     ++it;
  //   }
  // }
  pcl::toROSMsg(*surface_cloud, mesh_msg);
  mesh_msg.header.frame_id = "world";  // 设置点云的坐标系
  mesh_pub.publish(mesh_msg);
  cout << "send mesh finished" << endl;

  // send surface clusters cloud
  pcl::toROSMsg(*surface_clusters_cloud, surface_clusters_msg);
  surface_clusters_msg.header.frame_id = "world";  // 设置点云的坐标系
  surface_clusters_pub.publish(surface_clusters_msg);
  cout << "send surface clusters finished" << endl;

  // ROS_WARN("surface_cloud->size() = %d", surface_cloud->size());
  // ROS_WARN("surface_clusters_cloud->size() = %d", surface_clusters_cloud->size());


}


void SurfaceFinder::get_surface_points(const Eigen::Vector3d& cur_pos){
    ros::Time ts = ros::Time::now();
    surface_cloud->clear();
    points.clear();
    // 读取mesh data文件
    Eigen::MatrixXf result = readTextFile(mesh_data_base_dir);
    ROS_WARN("read mesh data time 1 = %f", (ros::Time::now() - ts).toSec());
    for (int i = 0; i < result.rows(); i = i + down_sample_) {

      pcl::PointXYZRGB point;
      point.x = result(i, 0);
      point.y = result(i, 1);
      point.z = result(i, 2);
      point.r = static_cast<uint8_t>(result(i, 6));
      point.g = static_cast<uint8_t>(result(i, 7));
      point.b = static_cast<uint8_t>(result(i, 8));
      surface_cloud->push_back(point);

      // filter points by distance
      Eigen::Vector3d pos = Eigen::Vector3d(result(i, 0), result(i, 1), result(i, 2));
      if ((pos - cur_pos).norm() > points_radius_) continue;
 
      // surface_cloud->push_back(point);
      PointWithUncertainty p;
      p.point = point;
      p.normal = Eigen::Vector3d(result(i, 3), result(i, 4), result(i, 5));
      p.uncertainty = result(i, 9);
      points.push_back(p);

    } 
    ROS_WARN("read mesh data time 2 = %f", (ros::Time::now() - ts).toSec());


}



Eigen::MatrixXf SurfaceFinder::readTextFile(const std::string& filename) {
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
        int cols = 10;
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




void SurfaceFinder::clusterPoints(vector<PointWithUncertainty>& points, vector<vector<PointWithUncertainty>>& clusters) {
    // 构建 KD 树
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr surface_points(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (const auto& p : points) {
        surface_points->push_back(p.point);
    }

    kdtree.setInputCloud(surface_points);
    // ROS_WARN("surface_cloud->size() = %d", surface_cloud->size());
    // ROS_WARN("points = %d", points.size());
    // 遍历每个点
    clusters.clear();
    while (points.size() > 100 and clusters.size() < cluster_num_) {
        std::vector<int> points_to_remove;  // 存储需要移除的点的索引

        // 找到当前不确定性最大的点作为起始点
        auto max_uncertainty_point = std::max_element(points.begin(), points.end(),
            [](const PointWithUncertainty& p1, const PointWithUncertainty& p2) {
                return p1.uncertainty < p2.uncertainty;
            });

        // 当前聚类
        std::vector<PointWithUncertainty> cluster;
        cluster.push_back(*max_uncertainty_point);

        // 从剩余点中移除起始点
        points.erase(max_uncertainty_point);

        // 更新 KD 树的输入云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        // ROS_WARN("points = %d", points.size());
        for (const auto& p : points) {
            remaining_cloud->push_back(p.point);
        }

        kdtree.setInputCloud(remaining_cloud);

        // 查询半径1范围内的点
        std::vector<int> point_indices;
        std::vector<float> point_distances;
        kdtree.radiusSearch(cluster[0].point, cluster_radius_, point_indices, point_distances);

        // 检查点的不确定性并添加到聚类中
        for (const auto& index : point_indices) {
            cluster.push_back(points[index]);
            points_to_remove.push_back(index);
        }

        // 将聚类添加到结果中
        clusters.push_back(cluster);

        // 将要删除的索引按降序排序，确保正确删除
        std::sort(points_to_remove.rbegin(), points_to_remove.rend());
        // 从后向前删除需要移除的点
        for (const auto& index : points_to_remove) {
            points.erase(points.begin() + index);
        }
    }

    
    surface_clusters.clear();
    int surface_clusters_points_num = 0;
    // surface_clusters
    for (size_t i = 0; i < clusters.size(); ++i) {
        SurfaceCluster surface_cluster;
        surface_cluster.id_ = i;
        for (size_t j = 0; j < clusters[i].size(); ++j) {
            Eigen::Vector3d point = clusters[i][j].point.getVector3fMap().cast<double>();
            surface_cluster.points_.push_back(point);
            surface_cluster.normals_.push_back(clusters[i][j].normal);
            surface_cluster.uncers_.push_back(clusters[i][j].uncertainty);
            surface_cluster.average_ += point;
            surface_clusters_points_num ++;
        }
        surface_cluster.average_ /= clusters[i].size();  // 计算均值
        surface_clusters.push_back(surface_cluster);
    }

    // // 打印每个聚类中的点的数量
    // cout << "clusters size = " << surface_clusters.size() << endl;
    // for (size_t i = 0; i < surface_clusters.size(); ++i) {
    //     std::cout << "SurfaceCluster " << i << ": " << surface_clusters[i].points_.size() << " points" << std::endl;
    // }

  // ROS_WARN("surface_clusters_points_num = %d", surface_clusters_points_num);

}

// color map
void SurfaceFinder::applyColorMapToClusters()
{   
    // 从surface_clusters中提取点实现以下功能，给出代码
    // 创建颜色映射
    cv::Mat color_map(surface_clusters.size(), 1, CV_8UC1);
    for (size_t i = 0; i < surface_clusters.size(); ++i) {
        // 计算归一化的值
        double normalized_value = 1-static_cast<double>(i) / surface_clusters.size();

        // 根据归一化值生成颜色索引
        uint8_t color_index = static_cast<uint8_t>(normalized_value * 255);

        // 设置颜色索引
        color_map.at<uint8_t>(i, 0) = color_index;
    }

    // 应用颜色映射
    cv::Mat color;
    cv::applyColorMap(color_map, color, cv::COLORMAP_JET);

    // 根据颜色映射设置聚类颜色
    surface_clusters_cloud->clear();
    pcl::PointXYZRGB point ;
    for (size_t i = 0; i < surface_clusters.size(); ++i) {
        uint8_t r = color.at<cv::Vec3b>(i, 0)[2];
        uint8_t g = color.at<cv::Vec3b>(i, 0)[1];
        uint8_t b = color.at<cv::Vec3b>(i, 0)[0];

        for (size_t j = 0; j < surface_clusters[i].points_.size(); ++j) {
            // 设置点的位置
            point.x = surface_clusters[i].points_[j][0];
            point.y = surface_clusters[i].points_[j][1];
            point.z = surface_clusters[i].points_[j][2];
            // 设置点的颜色
            point.r = r;
            point.g = g;
            point.b = b;

            // 添加点到点云
            surface_clusters_cloud->push_back(point);
        }
    }


}


void SurfaceFinder::test(const ros::TimerEvent& e){
    cout << "test" << endl;
    ros::Time ts = ros::Time::now();
    // // 聚类
    surface_clusters.clear();
    clusterPoints(points, points_clusters);
    cout << "t2 = " << ros::Time::now() - ts << endl;

    computeSurfaceclustersToVisit();
    cout << "t3 = " << ros::Time::now() - ts << endl;

    auto cur_pos = Eigen::Vector3d(0,1.5,-2);
    double cur_yaw = 0.0;
    double cur_pitch = 0.0;
    vector<int> indices;
    vector<Vector3d> surf_path;
    if (surface_clusters.size() > 1) {
        findGlobalTour(cur_pos, cur_yaw, cur_pitch, indices); 
        getPathForTour(cur_pos, indices, surf_path, surf_points_, surf_yaws_, surf_pitchs_);
    }
    cout << "t4 = " << ros::Time::now() - ts << endl;
    
    // 颜色映射
    applyColorMapToClusters();
    cout << "t5 = " << ros::Time::now() - ts << endl;

    // 发布mesh和clusters
    pubmeshCallback();

    // 画出视角
    drawFirstViews();
    // drawAllViews();

    // 画出全局路径
    drawTraj(surf_path);


}


void SurfaceFinder::computeSurfaceclustersToVisit(){

    cout << "[1]suraface clusters size = " << surface_clusters.size() << endl;
    // Try find viewpoints for each cluster and categorize them according to viewpoint number
    for (auto& surf_ctr : surface_clusters) {
        // Search viewpoints around frontier
        // cout << "tmp_ftr come " << endl;
        sampleViewpoints(surf_ctr);
    }


    // for循环删除没有viewpoints的cluster，从后往前删除，防止序号乱
    for (int i = surface_clusters.size() - 1; i >= 0; --i) {
        if (surface_clusters[i].viewpoints_.size() == 0) {
        surface_clusters.erase(surface_clusters.begin() + i);
        }
    }
    cout << "[filtered]suraface clusters size = " << surface_clusters.size() << endl;

}

void SurfaceFinder::sampleViewpoints(SurfaceCluster& surf_ctr) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Evaluate sample viewpoints on circles, find ones that cover most cells
  for (double rc = candidate_rmin_, dr = (candidate_rmax_ - candidate_rmin_) / candidate_rnum_; rc <= candidate_rmax_ + 1e-3; rc += dr){
    for (double phi = -M_PI/4.0; phi < M_PI/4.0; phi += candidate_dphi_) {
      for (double theta = -M_PI/4.0+M_PI/18; theta <=M_PI/4.0-M_PI/18; theta += M_PI/18.0){            // by zj

        // Vector3d sample_pos = surf_ctr.average_ + rc * Vector3d(cos(theta)*cos(phi),-sin(theta), cos(theta)*sin(phi));
        Vector3d sample_pos = surf_ctr.points_[0] + rc * Vector3d(cos(theta)*cos(phi),-sin(theta), cos(theta)*sin(phi));

        // add gauss noise to sample_pos
        // 创建一个随机数引擎
        std::normal_distribution<double> gaussian_dist(0.0, 0.1); // 均值为0，标准差为0.1，根据需要调整
        // 生成高斯噪声
        double noise_x = gaussian_dist(gen);
        double noise_y = gaussian_dist(gen);
        double noise_z = gaussian_dist(gen);
        sample_pos += Vector3d(noise_x, noise_y, noise_z);

        if(sample_pos[1] > 1.9 || sample_pos[1]<0.7 ) continue;  // by zj

        // Qualified viewpoint is in bounding box and in safe region
        if (!edt_env_->sdf_map_->isInBox(sample_pos) ||
            edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos))
          continue;
        
        if (edt_env_->sdf_map_->getDistance(sample_pos) < 0.3) continue;  

        // Eigen::Vector3d sample_pos_dir = sample_pos.normalized();
        // double sample_yaw = atan2(sample_pos_dir[0], sample_pos_dir[2]);
        // double sample_pitch = atan2(-sample_pos_dir[1], sqrt(sample_pos_dir[0]*sample_pos_dir[0]+sample_pos_dir[2]*sample_pos_dir[2]));
        // wrapYaw(sample_yaw);
        // cout << "[sample_yaw]: " << sample_yaw << ", [sample_pitch]: " << sample_pitch << endl;
        // if (sample_yaw < -M_PI_4 || sample_yaw > M_PI_4 || sample_pitch < -M_PI_4 || sample_pitch > M_PI_4)  continue;

        // cout << "[Accept !] " << "[sample_yaw]: " << sample_yaw << ", [sample_pitch]: " << sample_pitch << endl;        

        auto& cells = surf_ctr.points_;
        double avg_yaw = 0.0;
        double avg_pitch = 0.0;
        for (int i = 0; i < cells.size(); ++i) {
          Eigen::Vector3d dir = (cells[i] - sample_pos).normalized();
          avg_yaw = avg_yaw  + atan2(dir[0], dir[2]);  // by zj
          avg_pitch = avg_pitch  + atan2(-dir[1], sqrt(dir[0]*dir[0]+dir[2]*dir[2]));  // by zj
        }
        avg_yaw = avg_yaw / cells.size() ;  // by zj
        avg_pitch = avg_pitch / cells.size() ;  // by zj

        wrapYaw(avg_yaw);
  
        int visib_num = 0;
        double view_gain = 0.0;
        computerViewInfo(sample_pos, avg_pitch,avg_yaw, surf_ctr, visib_num, view_gain);
        if (visib_num >= min_visib_num_ ) {
          Viewpoint vp = { sample_pos, avg_pitch,avg_yaw, visib_num, view_gain};
          surf_ctr.viewpoints_.push_back(vp);
        }
       

      }
    }
    
    if (surf_ctr.viewpoints_.size() == 0) continue;
    // 从大到小排序
    // sort(surf_ctr.viewpoints_.begin(), surf_ctr.viewpoints_.end(), [](const Viewpoint& vp1, const Viewpoint& vp2) {
    //   return vp1.visib_num_ > vp2.visib_num_;
    // });
    sort(surf_ctr.viewpoints_.begin(), surf_ctr.viewpoints_.end(), [](const Viewpoint& vp1, const Viewpoint& vp2) {
      return vp1.gain_ > vp2.gain_;
    });
  }
 

}

bool SurfaceFinder::isNearUnknown(const Eigen::Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -vox_num; z <= vox_num; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

void SurfaceFinder::wrapYaw(double& yaw) {
  while (yaw < -M_PI)
    yaw += 2 * M_PI;
  while (yaw > M_PI)
    yaw -= 2 * M_PI;
}

void SurfaceFinder::computerViewInfo(const Eigen::Vector3d& pos,  const double& pitch, const double& yaw, const SurfaceCluster& surf_ctr, 
                         int& visib_num, double& view_gain) {
  percep_utils_->setPose(pos, pitch,yaw);
  Eigen::Vector3i idx;
  for (int i=0; i < surf_ctr.points_.size(); i++) {
    auto cell = surf_ctr.points_[i];
    auto uncer = surf_ctr.uncers_[i];
    auto normal = surf_ctr.normals_[i];
    // Check if frontier cell is inside FOV
    if (!percep_utils_->insideFOV(cell)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, pos);
    bool visib = true;
    int count = 0;
    int num = 0;
    while (raycaster_->nextId(idx)) {
      num ++;
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        if(num > 5 ) {
          count ++;
          break;
        };
      }
   }
   if (count < 1000) {
      visib_num += 1;
      // Compute view gain
      view_gain += uncer;
   }


 }

}


void SurfaceFinder::getFullCostMatrix(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch,
                         Eigen::MatrixXd& mat) {

    // Use Asymmetric TSP
    int dimen = surface_clusters.size();
    mat.resize(dimen + 1, dimen + 1);
    // std::cout << "mat size: " << mat.rows() << ", " << mat.cols() << std::endl;
    // Fill block for clusters
    int i = 1, j = 1;
    for (auto start_surf_ctr : surface_clusters) {
      for (auto end_surf_ctr : surface_clusters) {
        // std::cout << "(" << i << ", " << j << ")"
        // << ", ";
        vector<Vector3d> path;
        Viewpoint vi = start_surf_ctr.viewpoints_.front();
        Viewpoint vj = end_surf_ctr.viewpoints_.front();
        double cost_ij = ViewNode::computeCostNew(vi.pos_, vj.pos_, vi.yaw_, vj.yaw_, vi.pitch_, vj.pitch_, path);
        start_surf_ctr.costs_.push_back(cost_ij);
        start_surf_ctr.paths_.push_back(path);
        mat(i, j++) = cost_ij;
      }
      ++i;
      j = 1;
    }
    // std::cout << "" << std::endl;

    // Fill block from current state to clusters
    mat.leftCols<1>().setZero();
    for (auto surf_ctr : surface_clusters) {
      // std::cout << "(0, " << j << ")"
      // << ", ";
      Viewpoint vj = surf_ctr.viewpoints_.front();
      vector<Vector3d> path;
      mat(0, j++) =
          ViewNode::computeCostNew(cur_pos, vj.pos_, cur_yaw, vj.yaw_, cur_pitch, vj.pitch_, path);
    }
    // std::cout << "" << std::endl;
  
}


void SurfaceFinder::findGlobalTour(const Vector3d& cur_pos, const double& cur_yaw, const double& cur_pitch, vector<int>& indices) {
  
    auto t1 = ros::Time::now();
    // Get cost matrix for current state and clusters
    Eigen::MatrixXd cost_mat;
    getFullCostMatrix(cur_pos, cur_yaw, cur_pitch, cost_mat);
    // // cout cost_mat 的值
    // cout << "[surface]cost_mat = " << endl;
    // for (int i = 0; i < cost_mat.rows(); i++) {
    //     for (int j = 0; j < cost_mat.cols(); j++) {
    //         cout << cost_mat(i, j) << " ";
    //     }
    //     cout << endl;
    // }
  
  const int dimension = cost_mat.rows();
  ROS_WARN("surface dimension = %d", dimension);

  double mat_time = (ros::Time::now() - t1).toSec();
  t1 = ros::Time::now();

  // Write params and cost matrix to problem file
  ofstream prob_file(tsp_dir_ + "/single.tsp");
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
  solveTSPLKH((tsp_dir_ + "/single.par").c_str());

  // Read optimal tour from the tour section of result file
  ifstream res_file(tsp_dir_ + "/single.txt");
  string res;
  while (getline(res_file, res)) {
    // Go to tour section
    if (res.compare("TOUR_SECTION") == 0) break;
  }

  // Read path for ATSP formulation
  while (getline(res_file, res)) {
      // Read indices of frontiers in optimal tour
      int id = stoi(res);
      if (id == 1)  // Ignore the current state
         continue;
      if (id == -1) break;
      indices.push_back(id - 2);  // Idx of solver-2 == Idx of frontier
  }


  res_file.close();


  double tsp_time = (ros::Time::now() - t1).toSec();
  ROS_WARN("Cost mat: %lf, TSP: %lf", mat_time, tsp_time);



}

void SurfaceFinder::getPathForTour(const Vector3d& pos, const vector<int>& surf_ctr_ids, vector<Vector3d>& path,
               vector<Vector3d>& points, vector<double>& yaws, vector<double>& pitchs) {
  // Make an surf_ctr_indexer to access the frontier list easier
  vector<vector<SurfaceCluster>::iterator> surf_ctr_indexer;
  for (auto it = surface_clusters.begin(); it != surface_clusters.end(); ++it)
    surf_ctr_indexer.push_back(it);

  double dist = 0.0;
  // Compute the path from current pos to the first frontier
  vector<Vector3d> segment;
  dist = ViewNode::searchPath(pos, surf_ctr_indexer[surf_ctr_ids[0]]->viewpoints_.front().pos_, segment);
  if(dist == 5000 || dist > max_path_length_) return;
  path.insert(path.end(), segment.begin(), segment.end());
  points.push_back(surf_ctr_indexer[surf_ctr_ids[0]]->viewpoints_.front().pos_);
  yaws.push_back(surf_ctr_indexer[surf_ctr_ids[0]]->viewpoints_.front().yaw_);
  pitchs.push_back(surf_ctr_indexer[surf_ctr_ids[0]]->viewpoints_.front().pitch_);

  // Get paths of tour passing all clusters
  for (int i = 0; i <surf_ctr_ids.size() - 1; ++i) {
    // Move to path to next cluster
    vector<Vector3d> segment;
    auto p1 = surf_ctr_indexer[surf_ctr_ids[i]]->viewpoints_.front().pos_;
    auto p2 = surf_ctr_indexer[surf_ctr_ids[i + 1]]->viewpoints_.front().pos_;
    double path_length = ViewNode::searchPath(p1, p2, segment);
    if(path_length == 5000) break;
    dist += path_length;
    if (dist <= max_path_length_){
       path.insert(path.end(), segment.begin(), segment.end());
       points.push_back(p2);
       yaws.push_back(surf_ctr_indexer[surf_ctr_ids[i + 1]]->viewpoints_.front().yaw_);
       pitchs.push_back(surf_ctr_indexer[surf_ctr_ids[i + 1]]->viewpoints_.front().pitch_);
    }


  }


}


void SurfaceFinder::drawTraj(const vector<Vector3d>& plan_path){
  // 创建路径消息的可视化对象
  visualization_msgs::Marker traj_marker;
  ros::Time current_time = ros::Time::now();
  traj_marker.header.frame_id = "world";
  traj_marker.header.stamp = current_time;
  traj_marker.ns = "trajectory";
  traj_marker.id = 0;
  traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
  traj_marker.action = visualization_msgs::Marker::ADD;
  traj_marker.pose.orientation.w = 1.0;
  traj_marker.scale.x = 0.02; // 设置线条的宽度

  // 设置路径线条的颜色为红色
  traj_marker.color.r = 1.0;
  traj_marker.color.g = 0.0;
  traj_marker.color.b = 0.0;
  traj_marker.color.a = 1.0; // 不透明度

  // 设置路径线条的顶点
  for (int i = 0; i < plan_path.size(); i++) {
    geometry_msgs::Point point;
    point.x = plan_path[i][0];
    point.y = plan_path[i][1];
    point.z = plan_path[i][2];
    traj_marker.points.push_back(point);
  }

  // 发布可视化消息
  traj_pub.publish(traj_marker);

  // // 循环输出plan_path，plan_path是全局路径
  // cout << "surface plan_path = " << endl;
  // for (int i = 0; i < plan_path.size(); i++) {
  //     cout  << plan_path[i].transpose() << "->";
  // }
  // cout << endl;

}


void SurfaceFinder::drawFirstViews() {
  // 删除 MarkerArray 中的所有内容
  visualization_msgs::MarkerArray mk_array;
  mk_array.markers.clear();
  mk_array.markers.resize(0);
  views_vis_pub.publish(mk_array);

  // 发布新的内容
  int n = 0;
  for (auto surf_ctr : surface_clusters){
    // cout << n << ", viewpoint size = " << surf_ctr.viewpoints_.size() << ", surf_ctr average_ =" << surf_ctr.average_.transpose() << endl;
    if(surf_ctr.viewpoints_.size() == 0) continue;
    // cout << " viewpoint_[0] vis_num = " << surf_ctr.viewpoints_[0].visib_num_  << ", pos = " <<  surf_ctr.viewpoints_[0].pos_.transpose()
    // << ", pitch = " <<  surf_ctr.viewpoints_[0].pitch_ << ", yaw = " <<  surf_ctr.viewpoints_[0].yaw_ << endl; 
    Vector3d pos = surf_ctr.viewpoints_[0].pos_;
    double yaw = surf_ctr.viewpoints_[0].yaw_;
    double pitch = surf_ctr.viewpoints_[0].pitch_;
    // cout << "view = " << pos << yaw << endl;
    percep_utils_->setPose(pos, pitch, yaw);
    vector<Eigen::Vector3d> list1, list2;
    percep_utils_->getFOV(list1, list2);
    // cout <<"l1,l2 = "<<l1.size() <<l2.size() <<endl;
  

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = n;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = 1.0;
    mk.color.g = 0.0;
    mk.color.b = 0.0;
    mk.color.a = 1.0;
    mk.scale.x = 0.04;
    mk.scale.y = 0.04;
    mk.scale.z = 0.04;

    if (list1.size() == 0) return;

    // Pub new marker
    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i) {
      pt.x = list1[i](0);
      pt.y = list1[i](1);
      pt.z = list1[i](2);
      mk.points.push_back(pt);

      pt.x = list2[i](0);
      pt.y = list2[i](1);
      pt.z = list2[i](2);
      mk.points.push_back(pt);
    }
    mk.action = visualization_msgs::Marker::ADD;
    // if (n==0) 
    mk_array.markers.push_back(mk);
    n++;

  }
  views_vis_pub.publish(mk_array);
}

void SurfaceFinder::drawAllViews() {
  // 删除 MarkerArray 中的所有内容
  visualization_msgs::MarkerArray mk_array;
  mk_array.markers.clear();
  mk_array.markers.resize(0);
  views_vis_pub.publish(mk_array);
  ros::Duration(0.005).sleep();
  // for(int i=0; i < 10; i++){
  //   views_vis_pub.publish(mk_array);
  //   ros::Duration(0.02).sleep();
  // }

  // 发布新的内容
  int n = 0,n_i = 0;
  cout << "[2]suraface clusters size = " << surface_clusters.size() << endl;
  for (auto surf_ctr : surface_clusters){
    // if (n>0) break;
    cout << n << ", viewpoint size = " << surf_ctr.viewpoints_.size() << ", surf_ctr average_ =" << surf_ctr.average_.transpose() << endl;
    
    int N = min(10, int(surf_ctr.viewpoints_.size()));
    for (int i = 0; i< N;i++){
    //   cout << "      viewpoint vis_num = " << surf_ctr.viewpoints_[i].visib_num_  << ", pos = " <<  surf_ctr.viewpoints_[i].pos_.transpose()
    //   << ", pitch = " <<  surf_ctr.viewpoints_[i].pitch_  << ", yaw = " <<  surf_ctr.viewpoints_[i].yaw_ << endl; 
      Vector3d pos = surf_ctr.viewpoints_[i].pos_;
      double yaw = surf_ctr.viewpoints_[i].yaw_;
      double pitch = surf_ctr.viewpoints_[i].pitch_;
      // cout << "view = " << pos << yaw << endl;
      percep_utils_->setPose(pos, pitch, yaw);
      vector<Eigen::Vector3d> list1, list2;
      percep_utils_->getFOV(list1, list2);
      // cout <<"l1,l2 = "<<l1.size() <<l2.size() <<endl;
    

      visualization_msgs::Marker mk;
      mk.header.frame_id = "world";
      mk.header.stamp = ros::Time::now();
      mk.id = n_i;
      mk.ns = "current_pose";
      mk.type = visualization_msgs::Marker::LINE_LIST;
      mk.pose.orientation.x = 0.0;
      mk.pose.orientation.y = 0.0;
      mk.pose.orientation.z = 0.0;
      mk.pose.orientation.w = 1.0;
      mk.color.r = 1.0;
      mk.color.g = 0.0;
      mk.color.b = 0.0;
      mk.color.a = 1.0;
      mk.scale.x = 0.04;
      mk.scale.y = 0.04;
      mk.scale.z = 0.04;

      if (list1.size() == 0) return;

      // Pub new marker
      geometry_msgs::Point pt;
      for (int i = 0; i < int(list1.size()); ++i) {
        pt.x = list1[i](0);
        pt.y = list1[i](1);
        pt.z = list1[i](2);
        mk.points.push_back(pt);

        pt.x = list2[i](0);
        pt.y = list2[i](1);
        pt.z = list2[i](2);
        mk.points.push_back(pt);
      }
      mk.action = visualization_msgs::Marker::ADD;
      if (n==2) mk_array.markers.push_back(mk);
      n_i++;
    }
    n++;

  }
  cout << "[surface] mk_array size = " << mk_array.markers.size() << endl;
  if(mk_array.markers.size() > 0) views_vis_pub.publish(mk_array);
}


void SurfaceFinder::visualize(vector<Vector3d>& surface_path){
    // 颜色映射
    applyColorMapToClusters();
    // cout << "t5 = " << ros::Time::now() - ts << endl;
    
    // 发布mesh和clusters
    pubmeshCallback();

    // 画出视角
    drawFirstViews();
    // drawAllViews();

    // 画出全局路径
    drawTraj(surface_path);

}

void SurfaceFinder::drawSampleViews(vector<Vector3d> & sample_pos, vector<double> & sample_pitchs, 
                           vector<double> & sample_yaws, vector<double> & uncers){

  double uncer_max, uncer_min;
  Eigen::VectorXd uncer_vec = Eigen::VectorXd::Map(uncers.data(), uncers.size());
  uncer_max = uncer_vec.maxCoeff();
  uncer_min = uncer_vec.minCoeff();
  // normalize
  for(int i=0; i < uncers.size(); i++){
    uncers[i] = (uncers[i] - uncer_min) / (uncer_max - uncer_min)*255.0;
  }

  vector<Vector3d> uncer_colors;
  for(int i=0; i < uncers.size(); i++){
    // uncers to cv::COlormap_jet 
    cv::Mat uncer_mat = cv::Mat(1, 1, CV_8UC1, uncers[i]);
    cv::Mat uncer_jet;
    cv::applyColorMap(uncer_mat, uncer_jet, cv::COLORMAP_JET);

    Vector3d color;
    color(0) = uncer_jet.at<cv::Vec3b>(0,0)[0]/255.0;
    color(1) = uncer_jet.at<cv::Vec3b>(0,0)[1]/255.0;
    color(2) = uncer_jet.at<cv::Vec3b>(0,0)[2]/255.0;
    uncer_colors.push_back(color);
  }


  // uncer_colors sort
  vector<int> uncers_sort_idx;
  uncers_sort_idx.resize(uncers.size());
  iota(uncers_sort_idx.begin(), uncers_sort_idx.end(), 0);
  // form large to small
  sort(uncers_sort_idx.begin(), uncers_sort_idx.end(), 
    [&uncers](int i1, int i2) {return uncers[i1] > uncers[i2];});

  // cout uncer_sort
  for (int i = 0; i < uncers_sort_idx.size(); ++i)
  {
    uncers[uncers_sort_idx[i]] = uncers[uncers_sort_idx[i]]/255.0;
    // cout  << uncers[uncers_sort_idx[i]] << "; ";
    // cout << uncer_colors[uncers_sort_idx[i]] << "; ";
  }

  
  // 删除 MarkerArray 中的所有内容
  sample_views_mks.markers.clear();
  sample_views_mks.markers.resize(0);
  views_vis_pub.publish(sample_views_mks);
  // for(int i=0; i < 10; i++){
  //   path_views_vis_pub.publish(path_views_mks);
  //   ros::Duration(0.02).sleep();
  // }
  // 发布新的内容
  int n = 0;
  int N = sample_pos.size();
  // int N = 100;
  for(int k=0; k < N; k++){
    int index = uncers_sort_idx[k];
    // cout << index << ": " << uncers[index] << ", " << uncer_colors[index].transpose() << endl;

    Vector3d pos = sample_pos[index];
    double yaw = sample_yaws[index];
    double pitch = sample_pitchs[index];
    // cout << "view = " << pos << yaw << endl;
    percep_utils_->setPose(pos,pitch,yaw);
    vector<Eigen::Vector3d> list1, list2;
    percep_utils_->getFOV(list1, list2);
    // cout <<"l1,l2 = "<<l1.size() <<l2.size() <<endl;
  

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = n;
    mk.ns = "current_pose";
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;
    mk.color.r = uncer_colors[index][2];
    mk.color.g = uncer_colors[index][1];
    mk.color.b = uncer_colors[index][0];
    mk.color.a = 1.0;
    mk.scale.x = 0.04;
    mk.scale.y = 0.04;
    mk.scale.z = 0.04;

    if (list1.size() == 0) return;

    // Pub new marker
    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i) {
      pt.x = list1[i](0);
      pt.y = list1[i](1);
      pt.z = list1[i](2);
      mk.points.push_back(pt);

      pt.x = list2[i](0);
      pt.y = list2[i](1);
      pt.z = list2[i](2);
      mk.points.push_back(pt);
    }
    mk.action = visualization_msgs::Marker::ADD;
    sample_views_mks.markers.push_back(mk);
    n++;

  }

  views_vis_pub.publish(sample_views_mks);



}

}  // namespace fast_planner