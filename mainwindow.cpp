#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QPixmap>
#include <QImage>
#include <QVector>
#include <QQueue>

// #include <iostream>
#include <fstream>
#include <list>
#include <string>

#include <cmath>
#include <vector>
#include <algorithm>


// onnxruntime library
#include<onnxruntime_cxx_api.h>


const int targetWidth = 384;       // image scaled width
const int targetHeight = 384;      // image scaled height

// Define normalization parameters
std::array<float,3> image_mean = {0.48145466, 0.4578275, 0.40821073};
std::array<float,3> image_std = {0.26862954, 0.26130258, 0.27577711};


// Path to your ONNX model file
const char* modelVisionPath = "/home/tensor/caption_d/vision_model.onnx";
const char* modelTextPath = "/home/tensor/caption_d/text_decoder_model.onnx";

// Tokenizer file path
const char* TokenizerPath = "/home/tensor/caption_d/vocab.txt";

std::list<std::string> text_tock;




// load models amd sessions
Ort::RunOptions runOptions;
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeModel");
// Create a session options object
Ort::SessionOptions session_options;
// Create a session using the model path and session options to load vision model
Ort::Session vision_model_session{env, modelVisionPath, session_options};

Ort::Session text_model_session{env, modelTextPath, session_options};




std::string decode( std::vector<int>& intVector) {
    std::list<int> values_to_remove = {0, 100, 101, 102, 103};

    // Use erase-remove idiom with std::remove_if to remove specific values
    intVector.erase(std::remove_if(intVector.begin(), intVector.end(),
                                    [&values_to_remove](int value) {
                                        return std::find(values_to_remove.begin(), values_to_remove.end(), value) != values_to_remove.end();
                                    }
                                    ), intVector.end());

    std::string concatenatedString;

    for (const int& intValue : intVector) {
        // Assuming text_tock is some kind of container holding strings
        auto it = std::next(text_tock.begin(), intValue);
        concatenatedString += *it + " ";
    }

    // Iterate through each character and remove special characters
    std::string resultString;
    for (char c : concatenatedString) {
        if (std::isalnum(c) || std::isspace(c)) {
            // Keep alphanumeric and space characters
            resultString += c;
        }
    }

    return resultString;
}

std::list<std::string> get_tokens(){
    std::ifstream file(TokenizerPath);
    // Check if the file is open
    if (!file.is_open()) {
        qDebug() << "Error opening vocab text file!" ;
    }

    std::list<std::string> lines;
    // Read each line from the file and store it in the list
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Close the file
    file.close();
    return lines;
}


// Function to find the index of the maximum element in a std::array
template <size_t N>
size_t argmax(const std::array<float, N>& arr) {
    return std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
}


// Function to calculate softmax probabilities for a std::array
template <size_t N>
std::array<float, N> softmax(const std::vector<float>& output_logits) {
    // Your implementation of softmax function here
    std::array<float, N> result;

    // Find the maximum logit value
    float max_logit = *std::max_element(output_logits.begin(), output_logits.end());

    // Compute softmax probabilities
    float sum = 0.0f;
    for (std::size_t i = 0; i < N; ++i) {
        float exp_logit = std::exp(output_logits[i] - max_logit);
        result[i] = exp_logit;
        sum += exp_logit;
    }

    // Normalize to get probabilities
    for (std::size_t i = 0; i < N; ++i) {
        result[i] /= sum;
    }

    return result;
}



std::array<float,  1 * 3 * targetHeight * targetWidth> preprocess(QImage img){
    QImage scaledImage = img.scaled(targetWidth, targetHeight);
    scaledImage = scaledImage.convertToFormat(QImage::Format_RGB888);
    // Creating a 3D array
    std::vector<std::vector<std::vector<float>>> originalArray(targetHeight, std::vector<std::vector<float>>(targetWidth, std::vector<float>(3, 0)));
    // Transposing the array (swapping dimensions)
    std::vector<std::vector<std::vector<float>>> transposedArray(3, std::vector<std::vector<float>>(targetHeight, std::vector<float>(targetWidth, 0)));


    // Normalize image to mean and standard deviation
    for (int y = 0; y < scaledImage.height(); ++y){
        for (int x = 0; x < scaledImage.width(); ++x)
        {
            QColor pixelColor(scaledImage.pixel(x, y));
            float r = (pixelColor.redF() - image_mean[0]) / image_std[0];
            float g = (pixelColor.greenF() - image_mean[1]) / image_std[1];
            float b = (pixelColor.blueF() - image_mean[2]) / image_std[2];

            originalArray[x][y][0]=r;
            originalArray[x][y][1]=g;
            originalArray[x][y][2]=b;
        }

    }

    // adjust the array to model input shape
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < targetHeight; ++i) {
            for (int j = 0; j < targetWidth; ++j) {
                transposedArray[c][i][j] = originalArray[j][i][c];
            }
        }
    }

    // Convert to std::array
    std::array<float, 1 * 3 * targetHeight * targetWidth> input_array;

    size_t index = 0;
    for (const auto &outerVec : transposedArray) {
        for (const auto &middleVec : outerVec) {
            for (const auto &innerVal : middleVec) {
                input_array[index++] = innerVal;
            }
        }
    }
    //qDebug() << "pixel value"<< (input_array[65]);

    return input_array;


}


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    text_tock = get_tokens();
    qDebug() << "Hello to the world ";
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


QString file_name;
void MainWindow::on_pushButton_clicked()
{
     ui-> label -> setText("");
    file_name = QFileDialog::getOpenFileName(this, tr("Open File"), QDir::homePath(), tr("Images (*.png *.xpm *.jpg)"));
    if(!file_name.isEmpty()){
        //open prompt and display image
        QMessageBox::information(this, "...", file_name);
        QImage img(file_name);
        QPixmap pix = QPixmap::fromImage(img);

        int w = ui->label_pic->width();
        int h = ui->label_pic->height();

        ui->label_pic->setPixmap(pix.scaled(w,h,Qt::KeepAspectRatio));



    }

}


void MainWindow::on_predict_clicked()
{
    if(!file_name.isEmpty()){
        QImage img(file_name);


        std::vector<int64_t> inputShape{ 1, 3, 384, 384 };
        const std::array<int64_t, 3> outputShape = { 1, 577, 1024 };


        std::array<float, 1 * 3 * targetHeight * targetWidth> img_array = preprocess(img);
        std::array<float, 1*577*1024> results;



        // define Tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, img_array.data(), img_array.size(), inputShape.data(), inputShape.size());
        auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

        // define names
        Ort::AllocatorWithDefaultOptions ort_alloc;
        Ort::AllocatedStringPtr inputName = vision_model_session.GetInputNameAllocated(0, ort_alloc);
        Ort::AllocatedStringPtr outputName = vision_model_session.GetOutputNameAllocated(0, ort_alloc);



        const std::array<const char*, 1> inputNames = { inputName.get()};
        const std::array<const char*, 1> outputNames = { outputName.get()};
        inputName.release();
        outputName.release();


        try{
            vision_model_session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
        }
        catch (Ort::Exception){
            qDebug() << "Error on inference with vision model";
        }


        //#####################RUN TEXT MODEL############################
        Ort::AllocatorWithDefaultOptions allocator;
        size_t input_count = text_model_session.GetInputCount();
        std::vector<const char*> input_names = {"input_ids", "attention_mask", "encoder_hidden_states", "encoder_attention_mask"};
        std::vector<int64_t> input_dims;

        for (size_t i = 0; i < input_count; ++i) {

            input_dims = text_model_session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

        }

        const char* output_name = "2615";





        const char* input_names_array[] = { input_names[0], input_names[1], input_names[2], input_names[3] };


        std::vector<int> pred_index;
        pred_index.push_back(0);
        for(int m=1; m<22; m++){

            std::vector<int64_t> input_1_shape{1, m};  // Change the shape to 2D
            std::vector<int64_t> input_1(1,m);

            std::vector<int64_t> input_2_shape{1, m};
            std::vector<int64_t> input_2(1,m);

            for (int value : pred_index) {

               input_1.push_back(value);
               input_2.push_back(1);
            }
            input_1.erase(input_1.begin());
            input_2.erase(input_2.begin());


            auto input_tensor_1 = Ort::Value::CreateTensor<int64_t>(memory_info, input_1.data(), input_1.size(), input_1_shape.data(), input_1_shape.size());
            auto input_tensor_2 = Ort::Value::CreateTensor<int64_t>(memory_info, input_2.data(), input_2.size(), input_2_shape.data(), input_2_shape.size());


            std::vector<int64_t> input_3_shape{1, 577, 1024};
            std::array<float, 1* 577* 1024 > input_3 = results;
            auto input_tensor_3 = Ort::Value::CreateTensor<float>(memory_info, input_3.data(), input_3.size(), input_3_shape.data(), input_3_shape.size());


            std::vector<int64_t> input_4_shape{1, 577};
            std::array<int64_t, 1* 577 > input_4 ;
            input_4.fill(1);
            auto input_tensor_4 = Ort::Value::CreateTensor<int64_t>(memory_info, input_4.data(), input_4.size(), input_4_shape.data(), input_4_shape.size());


            const std::array<int64_t, 3> outputShape_t = { 1, m, 30524 };
            std::vector<float> results_tx(m * 30524);
            //std::array<float, 2*30524> results_tx;
            auto outputTensor_ = Ort::Value::CreateTensor<float>(memory_info, results_tx.data(), results_tx.size(), outputShape_t.data(), outputShape_t.size());

            text_model_session.Run(runOptions , input_names_array, &input_tensor_1, input_count, &output_name, &outputTensor_,1);


            //std::array<std::array<float, 30524>, 2> converted_array;
            std::vector<std::vector<float>> converted_array(m, std::vector<float>(30524));

            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < 30524; ++j) {
                    converted_array[i][j] = results_tx[i * 30524 + j];
                }
            }



            std::array<float, 1*30524> probabilities = softmax<1*30524>(converted_array.back());
            size_t pred_class = argmax(probabilities);
            pred_index.push_back(pred_class);
            // qDebug()<<pred_index;




        }

        std::string output = decode(pred_index);

        QString text = QString::fromStdString(output);
        ui-> label -> setText(text);
    }
}
