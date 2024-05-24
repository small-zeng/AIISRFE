#include <ros/ros.h>
#include <exploration_manager/test.h>
#include <httplib.h>
#include <Eigen/Dense>
#include <vector>
#include <torch/torch.h>
#include <cstdlib>


using namespace std;
using namespace Eigen;


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



Eigen::MatrixXf readTextFile(const std::string& filename) {
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


int main(int argc, char** argv) {
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;

    // 设置 CUDA_VISIBLE_DEVICES 环境变量为 "0,1,2,3"
    std::string devices = "0,1,2,3";
    std::string env_var_name = "CUDA_VISIBLE_DEVICES";
    std::string env_var_value = env_var_name + "=" + devices;
    int put_env_result = putenv(const_cast<char*>(env_var_value.c_str()));

    torch::Device device(torch::kCUDA, 2); // 设置默认的 CUDA 设备为 2

    string data_dir = "/mnt/dataset/zengjing/monosdf_planning/xyzg.txt";
    Eigen::MatrixXf result = readTextFile(data_dir).leftCols(4);
    result = result.block(0, 0, 8000, 4);
    // Select every 80th row
    int num_selected_rows = 8000 / 80;
    Eigen::MatrixXf selected_data(num_selected_rows, 4);
    for (int i = 0, j = 0; i < result.rows(); i += 80, j++) {
        selected_data.row(j) = result.block(i, 0, 1, 4);
    }
 
    // Convert Eigen matrix to torch::Tensor
    torch::Tensor data = torch::from_blob(selected_data.data(), {selected_data.rows(), selected_data.cols()});


    torch::manual_seed(1); // 设置随机种子
    const int num_samples = result.rows(); // 数据样本数量
    const int num_features = 4; // 数据特征数量

    cout << "generateGaussianData finished" << endl;

    // 将数据划分为训练集和测试集
    const int train_size = 80;
    torch::Tensor train_data = data.slice(0, 0, train_size); // Slice along dimension 0 (rows) to get the training data
    torch::Tensor train_features = train_data.slice(1, 0, num_features - 1); // Get the training data features
    torch::Tensor train_labels = train_data.slice(1, num_features - 1, num_features); // Get the training data labels

    torch::Tensor test_data = data.slice(0, train_size, num_samples); // Slice along dimension 0 (rows) to get the test data
    torch::Tensor test_features = test_data.slice(1, 0, num_features - 1); // Get the test data features
    torch::Tensor test_labels = test_data.slice(1, num_features - 1, num_features); // Get the test data labels

    std::cout << "data size: " << data.sizes() << std::endl;
    std::cout << "train_data size: " << train_data.sizes() << std::endl;
    std::cout << "test_data size: " << test_data.sizes() << std::endl;

    // 创建 MLP 模型
    MLP model;
    // 如果可用，将模型移动到 GPU 上
    if (torch::cuda::is_available()) {
        model.to(device);
        train_features = train_features.to(device); // 将训练数据的特征部分移动到 GPU 上
        train_labels = train_labels.to(device); // 将训练数据的标签部分移动到 GPU 上
        test_features = test_features.to(device); // 将测试数据的特征部分移动到 GPU 上
        test_labels = test_labels.to(device); // 将测试数据的标签部分移动到 GPU 上
    }

    // 定义损失函数和优化器
    torch::nn::MSELoss loss_func;
    torch::optim::Adam optimizer(model.parameters(), /*lr=*/0.001);

    // 训练网络
    ros::Time start = ros::Time::now();
    model.train();
    for (int epoch = 1; epoch <= 300; ++epoch) {
        torch::Tensor predictions = model.forward(train_features); // 前向传播计算预测值
        torch::Tensor loss = loss_func(predictions, train_labels); // 计算损失

        optimizer.zero_grad(); // 清零梯度
        loss.backward(); // 反向传播计算梯度
        optimizer.step(); // 更新参数

        std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }
    cout << "Training time: " << (ros::Time::now() - start).toSec() << endl;

    // 测试网络
    model.eval();
    torch::Tensor test_predictions = model.forward(test_features); // 前向传播计算测试预测值
    torch::Tensor test_loss = loss_func(test_predictions, test_labels); // 计算测试损失
    std::cout << "Test Loss: " << test_loss.item<float>() << std::endl;

    ros::spin(); // ROS node loop

    return 0;
}



