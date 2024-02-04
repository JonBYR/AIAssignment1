#include "Network.h"
double Network::totalSquaredError = 0; //give definition to the static variable
Network::Network(double wBias, double wOne, double wTwo, double wThree, string type) 
{
	weightBias = wBias;
	weightOne = wOne;
	weightTwo = wTwo;
	weightThree = wThree;
	inputBias = 1;
	netType = type; //initialise the weights for each network
}
void Network::setInputs(double one, double two, double three) 
{
	inputOne = one;
	inputTwo = two;
	inputThree = three; //set the inputs
}
void Network::setCases(double one, double two) 
{
	errorOne = one;
	errorTwo = two; //set the outputs (for calculating errors)
}
void Network::setNames(string bias, string wOne, string wTwo, string wThree) 
{
	nameBias = bias;
	nameOne = wOne;
	nameTwo = wTwo;
	nameThree = wThree; //sets the names of the weights
}
void Network::forwardStep() 
{
	forward = ((weightBias * inputBias) + (weightOne * inputOne) + (weightTwo * inputTwo) + (weightThree * inputThree));
	//calculates the dot product of the inputs multiplied by the weights added together
}
double Network::Sigmoid() 
{
	if (netType == "Net7" || netType == "Net8") return 0; //these do not use sigmoid
	else 
	{
		sig = 1 / (1 + exp(-forward));
		return sig;
		//calculates the sigmoid activation function that is returned to be used in nets 7 and 8
	}
}
double Network::calculateError() 
{
	if (netType == "Net7")
	{
		error = errorOne - forward;
		totalSquaredError += pow(error, 2); //accumulates the output errors in a static variable
		return error; //returns the error to be used when calculating hidden errors
	}
	else if (netType == "Net8")
	{
		error = errorTwo - forward;
		totalSquaredError += pow(error, 2);
		return error;
	}
}
void Network::calculateError(double weightA, double weightB, double errOne, double errTwo) //function overloading for the output and hidden layer
{
	if (netType != "Net7" || netType != "Net8")
	{
		error = sig * (1 - sig) * ((weightA * errOne) + (weightB * errTwo));
	}
	//formula for hidden layers;
}
void Network::updateWeights() 
{
	weightBias = weightBias + (LEARNINGRATE * inputBias * error);
	weightOne = weightOne + (LEARNINGRATE * inputOne * error);
	weightTwo = weightTwo + (LEARNINGRATE * inputTwo * error);
	weightThree = weightThree + (LEARNINGRATE * inputThree * error);
	//each weight is updated by summing the initial weight with learning rate * corresponding input and the networks error
}
double Network::getWeight(int net) //accessor needed to calculate the errors in the hidden layer
{
	if (net == 4) return weightOne;
	else if (net == 5) return weightTwo;
	else if (net == 6) return weightThree;
}
void Network::outputWeights() 
{
	cout << netType << ":" << endl;
	cout << nameBias << ": " << weightBias << endl;
	cout << nameOne << ": " << weightOne << endl;
	cout << nameTwo << ": " << weightTwo << endl;
	cout << nameThree << ": " << weightThree << endl; //used to output the weights for each network
}
double Network::SoftMax(double otherOutput) 
{
	return exp(forward) / (exp(forward) + exp(otherOutput)); //softmax for calculating probability distribution
}
double Network::getOutput() //accessor needed for softmax
{
	return forward;
}
double Network::returnTotalError() //static method used to return the average for each test
{
	return 0.5 * Network::totalSquaredError;
}