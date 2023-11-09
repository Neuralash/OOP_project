#include "../include/NeuralNetwork.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

int main() {
    
    ANNConfig config;
    config.topology = {3, 2, 2};  
    config.learningRate = 0.05;
    config.momentum = 1.0;
    config.bias = 1.0;
    config.hActivation = A_RELU;
    config.oActivation = A_SIGM;
    config.weightsFile = "D:/ann/data/a2.json";  
    NeuralNetwork neuralNetwork(config);

    
    neuralNetwork.loadWeights(config.weightsFile);

    // Provide new input data for prediction
    std::vector<double> input = {1, 0, 1};  

    // Set the input data
    neuralNetwork.setCurrentInput(input);

    // Perform the feedforward process
    neuralNetwork.feedForward();

    // Get the activated values of the output layer
    std::vector<double> output = neuralNetwork.getActivatedVals(neuralNetwork.topologySize - 1);

    
    
    int predictedClass = (output[0] > output[1]) ? 1 : 0;

    // Print the predicted class
    std::cout << "Predicted Class: " << predictedClass << std::endl;

    return 0;
}
