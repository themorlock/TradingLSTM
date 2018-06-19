#include <cstdlib>
#include <vector>
#include <cmath>
#include "LSTM.h"
#include "Matrix.h"

LSTM::LSTM(unsigned const inputSize, unsigned const hiddenSize){
	this->inputSize = inputSize;
	this->hiddenSize = hiddenSize;

	aInputWeights = Matrix(hiddenSize, inputSize, randomWeight);
	aRecurrentWeights = Matrix(hiddenSize, hiddenSize, randomWeight);
	aBiases = Matrix(hiddenSize, 1, 1.0);

	iInputWeights = Matrix(hiddenSize, inputSize, randomWeight);
	iRecurrentWeights = Matrix(hiddenSize, hiddenSize, randomWeight);
	iBiases = Matrix(hiddenSize, 1, 1.0);

	fInputWeights = Matrix(hiddenSize, inputSize, randomWeight);
	fRecurrentWeights = Matrix(hiddenSize, hiddenSize, randomWeight);
	fBiases = Matrix(hiddenSize, 1, 1.0);

	oInputWeights = Matrix(hiddenSize, inputSize, randomWeight);
	oRecurrentWeights = Matrix(hiddenSize, hiddenSize, randomWeight);
	oBiases = Matrix(hiddenSize, 1, 1.0);
}

void LSTM::forwardPropogate(std::vector<Matrix> const &inputs){
	this->in = inputs;
	a.clear();
	a.resize(in.size());
	i.clear();
	i.resize(in.size());
	f.clear();
	f.resize(in.size());
	o.clear();
	o.resize(in.size());
	state.resize(in.size());
	out.resize(in.size());
	for(unsigned t = 0; t < in.size(); ++t){
		if(t == 0){
			a[t] = matrixFunction((aInputWeights ^ in[t]) + aBiases, inputActivationFunction);
			i[t] = matrixFunction((iInputWeights ^ in[t]) + iBiases, inputGateFunction);
			f[t] = matrixFunction((fInputWeights ^ in[t]) + fBiases, forgetGateFunction);
			o[t] = matrixFunction((oInputWeights ^ in[t]) + oBiases, outputGateFunction);
			state[t] = (a[t] * i[t]);
			out[t] = matrixFunction(state[t], outputActivationFunction) * o[t];
			continue;
		}
		a[t] = matrixFunction((aInputWeights ^ in[t]) + (aRecurrentWeights ^ out[t - 1]) + aBiases, inputActivationFunction);
		i[t] = matrixFunction((iInputWeights ^ in[t]) + (iRecurrentWeights ^ out[t - 1]) + iBiases, inputGateFunction);
		f[t] = matrixFunction((fInputWeights ^ in[t]) + (fRecurrentWeights ^ out[t - 1]) + fBiases, forgetGateFunction);
		o[t] = matrixFunction((oInputWeights ^ in[t]) + (oRecurrentWeights ^ out[t - 1]) + oBiases, outputGateFunction);
		state[t] = (a[t] * i[t]) + (f[t] * state[t - 1]);
		out[t] = matrixFunction(state[t], outputActivationFunction) * o[t];
	}
}

void LSTM::backwardPropogate(std::vector<Matrix> const &targets){
	aGradient.clear();
	aGradient.resize(targets.size());
	iGradient.clear();
	iGradient.resize(targets.size());
	fGradient.clear();
	fGradient.resize(targets.size());
	oGradient.clear();
	oGradient.resize(targets.size());
	inGradient.clear();
	inGradient.resize(targets.size());
	stateGradient.clear();
	stateGradient.resize(targets.size());
	outGradient.clear();
	outGradient.resize(targets.size());
	outDelta.clear();
	outDelta.resize(targets.size());
	for(unsigned t = targets.size() - 1; t > 0; --t){
		if(t == targets.size() - 1){
			outGradient[t] = (out[t] - targets[t]);
			stateGradient[t] = outGradient[t] * o[t] * (1 - (matrixFunction(state[t], stateActivationFunction) * matrixFunction(state[t], stateActivationFunction)));
			aGradient[t] = stateGradient[t] * i[t] * (1 - (a[t] * a[t]));
			iGradient[t] = stateGradient[t] * a[t] * i[t] * (1 - i[t]);
			fGradient[t] = stateGradient[t] * state[t - 1] * f[t] * (1 - f[t]);
			oGradient[t] = outGradient[t] * matrixFunction(state[t], stateActivationFunction) * o[t] * (1 - o[t]);
			inGradient[t] = ~link(4, aInputWeights, iInputWeights, fInputWeights, oInputWeights) ^ link(4, aGradient, iGradient, fGradient, oGradient);
			outDelta[t - 1] = ~link(4, aRecurrentWeights, iRecurrentWeights, fRecurrentWeights, oRecurrentWeights) ^ link(4, aGradient, iGradient, fGradient, oGradient);
			continue;
		}
		if(t < targets.size() - 1 && t > 0){
			outGradient[t] = (out[t] - targets[t]) + outDelta[t];
			stateGradient[t] = outGradient[t] * o[t] * (1 - (matrixFunction(state[t], stateActivationFunction) * matrixFunction(state[t], stateActivationFunction))) + stateGradient[t + 1] * f[t + 1];
			aGradient[t] = stateGradient[t] * i[t] * (1 - (a[t] * a[t]));
			iGradient[t] = stateGradient[t] * a[t] * i[t] * (1 - i[t]);
			fGradient[t] = stateGradient[t] * state[t - 1] * f[t] * (1 - f[t]);
			oGradient[t] = outGradient[t] * matrixFunction(state[t], stateActivationFunction) * o[t] * (1 - o[t]);
			inGradient[t] = ~link(4, aInputWeights, iInputWeights, fInputWeights, oInputWeights) ^ link(4, aGradient, iGradient, fGradient, oGradient);
			outDelta[t - 1] = ~link(4, aRecurrentWeights, iRecurrentWeights, fRecurrentWeights, oRecurrentWeights) ^ link(4, aGradient, iGradient, fGradient, oGradient);
			continue;
		}
	}
	unsigned t = 0;
	outGradient[t] = (out[t] - targets[t]) + outDelta[t];
	stateGradient[t] = outGradient[t] * o[t] * (1 - (matrixFunction(state[t], stateActivationFunction) * matrixFunction(state[t], stateActivationFunction))) + stateGradient[t + 1] * f[t + 1];
	aGradient[t] = stateGradient[t] * i[t] * (1 - (a[t] * a[t]));
	iGradient[t] = stateGradient[t] * a[t] * i[t] * (1 - i[t]);
	fGradient[t] = Matrix(hiddenSize, 1);
	oGradient[t] = outGradient[t] * matrixFunction(state[t], stateActivationFunction) * o[t] * (1 - o[t]);
	inGradient[t] = ~link(4, aInputWeights, iInputWeights, fInputWeights, oInputWeights) ^ link(4, aGradient, iGradient, fGradient, oGradient);
}

double randomWeight(){
	return rand() / (double) RAND_MAX;
}

double inputActivationFunction(double x){
	return tanh(x);
}
double inputGateFunction(double x){
	return 1 / (1 + exp(-x));
}
double forgetGateFunction(double x){
	return 1 / (1 + exp(-x));
}
double outputGateFunction(double x){
	return 1 / (1 + exp(-x));
}
double outputActivationFunction(double x){
	return tanh(x);
}
double stateActivationFunction(double x){
	return tanh(x);
}