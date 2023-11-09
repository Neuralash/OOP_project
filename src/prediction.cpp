#include "../include/NeuralNetwork.hpp"
#include <fstream>
#include <sstream>

int main() {
  ANNConfig config;
  config.topology = {784, 16, 16, 10};
  config.learningRate = 0.05;
  config.momentum = 1.0;
  config.bias = 1.0;
  config.hActivation = A_RELU;
  config.oActivation = A_SIGM;
  config.weightsFile = "data/a1.json";  // Replace with the path to your weights file

  NeuralNetwork neuralNetwork(config);

  // Load the trained weights
  neuralNetwork.loadWeights(config.weightsFile);

  // Create a vector to store the input data
  vector<double> input;

  // Replace "input_data.csv" with the path to your input .csv file
  const string inputCsvFile = "D:/ann/data/input_data.csv";

  // Read the input data from the .csv file
  ifstream inputFile(inputCsvFile);
  if (inputFile.is_open()) {
    string line;
    if (getline(inputFile, line)) {
      istringstream iss(line);
      string token;
      while (getline(iss, token, ',')) {
        double value = stod(token);
        input.push_back(value);
      }
    }
    inputFile.close();
  } else {
    cout << "Failed to open input .csv file." << endl;
    return 1;
  }

  // Set the input data
  neuralNetwork.setCurrentInput(input);

  // Perform the feedforward process
  neuralNetwork.feedForward();

  // Get the activated values of the output layer
  vector<double> output = neuralNetwork.getActivatedVals(neuralNetwork.topologySize - 1);

  // Find the digit with the highest activation value (your prediction)
  int predictedDigit = distance(output.begin(), max_element(output.begin(), output.end()));

  cout << "Predicted Digit: " << predictedDigit << endl;

  return 0;
}
