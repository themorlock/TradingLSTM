#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "Matrix.h"
#include "LSTM.h"

int main(){

	LSTM net(2, 4);

	std::vector<Matrix> inputs;
	std::vector<Matrix> targets;
	for(unsigned i = 0; i < 5; ++i){
		inputs.push_back(Matrix(2, 1));
		targets.push_back(Matrix(4, 1));
		for(unsigned j = 0; j < 2; ++j){
			inputs.back().get(j, 0) = rand() / (double) RAND_MAX;
		}
		for(unsigned k = 0; k < 4; ++k){
			targets.back().get(k, 0) = rand() / (double) RAND_MAX;
		}
	}
	net.forwardPropogate(inputs);
	net.backwardPropogate(targets);
	//std::cout << "asd" << std::endl;
	//	std::getchar();
}