#ifndef LSTM_H_
#define LSTM_H_

#include <cstdlib>
#include <vector>
#include <cmath>
#include "Matrix.h"

class LSTM{
public:
	LSTM(unsigned const inputSize, unsigned const hiddenSize);

	void forwardPropogate(std::vector<Matrix> const &inputs);
	void backwardPropogate(std::vector<Matrix> const &targets);
private:
	unsigned inputSize;
	unsigned hiddenSize;

	std::vector<Matrix> a;
	std::vector<Matrix> aGradient;
	Matrix aInputWeights;
	Matrix aRecurrentWeights;
	Matrix aBiases;

	std::vector<Matrix> i;
	std::vector<Matrix> iGradient;
	Matrix iInputWeights;
	Matrix iRecurrentWeights;
	Matrix iBiases;

	std::vector<Matrix> f;
	std::vector<Matrix> fGradient;
	Matrix fInputWeights;
	Matrix fRecurrentWeights;
	Matrix fBiases;

	std::vector<Matrix> o;
	std::vector<Matrix> oGradient;
	Matrix oInputWeights;
	Matrix oRecurrentWeights;
	Matrix oBiases;

	std::vector<Matrix> in;
	std::vector<Matrix> inGradient;
	std::vector<Matrix> state;
	std::vector<Matrix> stateGradient;
	std::vector<Matrix> out;
	std::vector<Matrix> outGradient;
	std::vector<Matrix> outDelta;
};

double randomWeight();

double inputActivationFunction(double x);
double inputGateFunction(double x);
double forgetGateFunction(double x);
double outputGateFunction(double x);
double outputActivationFunction(double x);
double stateActivationFunction(double x);

#endif