// AIAssignment1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include "Network.h"
#include <string>
#include <vector>
using namespace std;

int main()
{
    int epoch = 0;
    Network net4(0.9, 0.74, 0.8, 0.35, "Net4");
    Network net5(0.45, 0.13, 0.4, 0.97, "Net5");
    Network net6(0.36, 0.68, 0.1, 0.96, "Net6");
    Network net7(0.98, 0.35, 0.5, 0.9, "Net7");
    Network net8(0.92, 0.8, 0.13, 0.8, "Net8"); //initialise the networks and weights
    net4.setNames("Weight 04", "Weight 14", "Weight 24", "Weight 34");
    net5.setNames("Weight 05", "Weight 15", "Weight 25", "Weight 35");
    net6.setNames("Weight 06", "Weight 16", "Weight 26", "Weight 36");
    net7.setNames("Weight 07", "Weight 47", "Weight 57", "Weight 67");
    net8.setNames("Weight 08", "Weight 48", "Weight 58", "Weight 68");
    vector<Network*> networks = { &net4, &net5, &net6, &net7, &net8 }; //pointers so that any changes done outside the vector apply to objects in the vector
    cout << "Initial epoch: "<< endl;
    for (int i = 0; i < networks.size(); i++) networks[i]->outputWeights();
    cout << endl;
    double inputOne;
    double inputTwo;
    double inputThree;
    double inputFour;
    double inputFive;
    double average = 0;
    string line;
    string delim1 = " ";
    string delim2 = "\t"; //delimiters to split the inputs and output values
    remove("data-CMP2020M-item1-errors.txt"); //removes the file if it already exists
    ifstream trainFile("data-CMP2020M-item1-train.txt"); //opens the file for training data
    ofstream errorData("data-CMP2020M-item1-errors.txt"); //opens the file for writing the output errors
    for (epoch = 0; epoch < 1; epoch++) //epoch is the number of times for training
    {
        while (getline(trainFile, line)) //looks at each line in the file
        {
            vector<string> numbers; //inputs
            string inputs = line.substr(0, line.find(delim2)); //splits the line into the inputs and desired outputs
            line.erase(0, line.find(delim2) + delim2.length());
            string outputs = line;
            size_t pos = 0;
            while ((pos = inputs.find(delim1)) != std::string::npos)
            {
                numbers.push_back(inputs.substr(0, pos)); //split the inputs
                inputs.erase(0, pos + delim1.length());
            }
            inputOne = stod(numbers[0]);
            inputTwo = stod(numbers[1]);
            inputThree = stod(inputs);
            inputFour = stod(outputs.substr(0, outputs.find(delim1)));
            outputs.erase(0, outputs.find(delim1) + delim1.length()); //split the outputs
            inputFive = stod(outputs);
            net4.setInputs(inputOne, inputTwo, inputThree);
            net5.setInputs(inputOne, inputTwo, inputThree);
            net6.setInputs(inputOne, inputTwo, inputThree); //set the inputs for each network
            net4.forwardStep();
            net5.forwardStep();
            net6.forwardStep(); //perform the forward step on nets 4-6
            inputOne = net4.Sigmoid();
            inputTwo = net5.Sigmoid();
            inputThree = net6.Sigmoid(); //execute the sigmoid for nets 4-6
            net7.setInputs(inputOne, inputTwo, inputThree);
            net8.setInputs(inputOne, inputTwo, inputThree); //initilise these sigmoids as the inputs for the other two networks
            net4.setCases(inputFour, inputFive);
            net5.setCases(inputFour, inputFive);
            net6.setCases(inputFour, inputFive);
            net7.setCases(inputFour, inputFive); //set the outputs for all networks
            net8.setCases(inputFour, inputFive);
            net7.forwardStep();
            net8.forwardStep();
            double errOne = net7.calculateError();
            double errTwo = net8.calculateError(); //returns the error for nets 7 and 8 so that it can be used to calculate the hidden errors
            net4.calculateError(net7.getWeight(4), net8.getWeight(4), errOne, errTwo);
            net5.calculateError(net7.getWeight(5), net8.getWeight(5), errOne, errTwo);
            net6.calculateError(net7.getWeight(6), net8.getWeight(6), errOne, errTwo); //calculate errors in the hidden layers
            net4.updateWeights();
            net5.updateWeights();
            net6.updateWeights();
            net7.updateWeights();
            net8.updateWeights(); //update the weights for each network
            average += Network::returnTotalError(); //takes the Squared Error for each test and multiplies it by a half, which is then accumulated in average
            Network::totalSquaredError = 0; //reinitilises to 0 for the next training data
        }
        if (epoch <= 9) //should we reach the 10th iteration
        {
            cout << "Epoch: " << epoch + 1 << endl;
            for (int j = 0; j < networks.size(); j++) networks[j]->outputWeights();
            cout << endl;
        }
        errorData << average << endl; //writes the error data for the epoch to the error file which is a summation of all mean squared errors
        average = 0; //reinitialises to 0 for the next epoch
        trainFile.clear();
        trainFile.seekg(0); //clears the end of file flag in testFile and then seeks the start of the file again for the next epoch
    }
    trainFile.close();
    errorData.close(); //closes the files
    ifstream testFile("data-CMP2020M-item1-test.txt");
    string testLine;
    vector<string> unseenVector;
    while (getline(testFile, testLine)) 
    {
        string inputs = testLine.substr(0, testLine.find(delim2)); 
        testLine.erase(0, testLine.find(delim2) + delim2.length());
        size_t pos = 0;
        while ((pos = inputs.find(delim1)) != std::string::npos)
        {
            unseenVector.push_back(inputs.substr(0, pos)); 
            inputs.erase(0, pos + delim1.length()); //splits the inputs the same way as the train data
        }
        unseenVector.push_back(inputs);
    }
    testFile.close();
    for (int k = 0; k < 3; k++)
    {
        networks[k]->setInputs(stod(unseenVector[0]), stod(unseenVector[1]), stod(unseenVector[2]));
        networks[k]->forwardStep();
    }
    inputOne = net4.Sigmoid();
    inputTwo = net5.Sigmoid();
    inputThree = net6.Sigmoid();
    for (int k = 3; k < 5; k++)
    {
        networks[k]->setInputs(inputOne, inputTwo, inputThree);
        networks[k]->forwardStep(); //performs the same operations as it does when training to get the outputs
        cout << networks[k]->getOutput() << endl;
    }
    cout << "Softmax for Node 7: " << net7.SoftMax(net8.getOutput()) << endl;
    cout << "Softmax for Node 8: " << net8.SoftMax(net7.getOutput()) << endl; //gets the probability distruption for the outputs via softmax
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
