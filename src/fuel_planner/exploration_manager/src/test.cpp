#include <ros/ros.h>
#include <exploration_manager/test.h>
#include <httplib.h>
#include <Eigen/Dense>
#include <vector>
#include <jsoncpp/json/json.h>

using namespace std;
using namespace Eigen;

vector<double> convertJsonToVector(const Json::Value& jsonData) {
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

vector<double> get_uncertainty(const vector<Eigen::Vector3d>& locations, const vector<double>& us, const vector<double>& vs){
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
    httplib::Client client("10.15.198.53", 7000);
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




int main(int argc, char** argv) {

  ros::init(argc, argv, "test");
  ros::NodeHandle nh("~");

//   ros::Duration(1.0).sleep();

  cout << "test" << endl;

  vector<Eigen::Vector3d> locations;
  vector<double> us;
  vector<double> vs;
  locations.push_back(Vector3d(1,2,3));
  locations.push_back(Vector3d(4,5,6));
  us.push_back(1);
  us.push_back(2);
  vs.push_back(2);
  vs.push_back(3);
  ros::Time start_time = ros::Time::now();
  vector<double> uncers = get_uncertainty(locations,us,vs);
  ros::Time end_time = ros::Time::now();
  cout << "time = " << (end_time - start_time).toSec() << endl;
  cout << "uncers = ";
  for(int i=0; i<uncers.size(); i++){
	  cout << uncers[i] << ", ";
  }
  cout << endl;

  ros::spin();


  return 0;
}