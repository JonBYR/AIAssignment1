#pragma once
#include <iostream>
#include <cmath>
#include <string>
using namespace std;
class Network
{
public:
	Network(double wBias, double wOne, double wTwo, double wThree, string type);
	void setInputs(double one, double two, double three);
	void setCases(double one, double two);
	void setNames(string bias, string wOne, string wTwo, string wThree);
	void forwardStep();
	void outputWeights();
	double Sigmoid();
	double SoftMax(double otherOutput);
	double getOutput();
	double calculateError();
	void calculateError(double weightA, double weightB, double errOne, double errTwo);
	void updateWeights();
	double getWeight(int net);
	static double returnTotalError(); //static as we need to use a static variable in this method
	static double totalSquaredError; //static as we need to accumulate the output error values across all networks (for each test) 
private:
	double weightBias;
	double weightOne;
	double weightTwo;
	double weightThree;
	double forward;
	double inputOne;
	double inputTwo;
	double inputThree;
	double inputBias;
	double error;
	double sig;
	double errorOne;
	double errorTwo;
	string netType;
	string nameBias;
	string nameOne;
	string nameTwo;
	string nameThree;
	const double LEARNINGRATE = 0.1; //learning rate is always constant
};

