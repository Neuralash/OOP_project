#include "../../include/NeuralNetwork.hpp"

void NeuralNetwork::saveWeights(string filename) {
  json j  = {};

  vector< vector< vector<double> > > weightSet;

  for(int i = 0; i < this->weightMatrices.size(); i++) {
    weightSet.push_back(this->weightMatrices.at(i)->getValues());
  }

  j["weights"]      = weightSet;
  j["topology"]     = this->topology;
  j["learningRate"] = this->learningRate;
  j["momentum"]     = this->momentum;
  j["bias"]         = this->bias;

  std::ofstream o(filename);
  o << std::setw(4) << j << endl;
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
  this->input = input;

  for(int i = 0; i < input.size(); i++) {
    this->layers.at(0)->setVal(i, input.at(i));
  }
}







// In NeuralNetwork.cpp
// In NeuralNetwork.cpp
NeuralNetwork::NeuralNetwork(ANNConfig config) {
    this->topology = config.topology;
    this->learningRate = config.learningRate;
    this->momentum = config.momentum;
    this->bias = config.bias;

    // Initialize layers and weight matrices
    for (int i = 0; i < topology.size(); i++) {
        layers.push_back(new Layer(topology[i]));
    }

    for (int i = 1; i < topology.size(); i++) {
        // Create an instance of your Matrix class for weight matrices
        Matrix* weightMatrixObject = new Matrix(topology[i], topology[i - 1], true); // Pass true for random initialization

        weightMatrices.push_back(weightMatrixObject);
    }
}
