#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdarg>
#include "Matrix.h"

Matrix::Matrix(const unsigned rows, const unsigned cols){
	for(unsigned i = 0; i < rows; ++i){
		m.push_back(std::vector<double>());
		for(unsigned j = 0; j < cols; ++j){
			m.back().push_back(0);
		}
	}
}

Matrix::Matrix(const unsigned rows, const unsigned cols, double const &num){
	for(unsigned i = 0; i < rows; ++i){
		m.push_back(std::vector<double>());
		for(unsigned j = 0; j < cols; ++j){
			m.back().push_back(num);
		}
	}
}

Matrix::Matrix(const unsigned rows, const unsigned cols, double (*initializer)()){	
	for(unsigned i = 0; i < rows; ++i){
		m.push_back(std::vector<double>());
		for(unsigned j = 0; j < cols; ++j){
			m.back().push_back(initializer());
		}
	}
}

Matrix::Matrix(){
	for(unsigned i = 0; i < 1; ++i){
		m.push_back(std::vector<double>());
		for(unsigned j = 0; j < 1; ++j){
			m.back().push_back(0);
		}
	}
}

Matrix::Matrix(std::vector<double> const &v){
	m.push_back(v);
}

Matrix::Matrix(std::vector<std::vector<double> > const &v){
	m = v;
}

double& Matrix::get(const unsigned row, const unsigned col){
	return m[row][col];
}

double Matrix::get(const unsigned row, const unsigned col) const{
	return m[row][col];
}

Matrix Matrix::getRow(const unsigned row){
	Matrix result(m[row]);
	return result;
}

double Matrix::set(const unsigned row, const unsigned col, const unsigned val){
	double oldVal = m[row][col];
	m[row][col] = val;
	return oldVal;
}

double Matrix::rows() const{
	return m.size();
}

double Matrix::cols() const{
	return m[0].size();
}

std::ostream& operator<<(std::ostream& os, const Matrix &m){
	for(unsigned i = 0; i < m.rows(); ++i){
		for(unsigned j = 0; j < m.cols(); ++j){
			os << std::left << std::setw(2) << m.get(i, j) << (j < m.cols() - 1 ? " " : (i < m.rows() - 1 ? "\n" : "")) << std::flush;
		}
	}
	return os;
}

Matrix operator^(Matrix const &m1, Matrix const &m2){
	Matrix result(m1.rows(), m2.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			double sum = 0;
			for(unsigned k = 0; k < m2.rows(); ++k){
				sum += m1.get(i, k) * m2.get(k, j);
			}
			result.get(i, j) = sum;
		}
	}
	return result;
}

Matrix operator%(Matrix const &m1, Matrix const &m2){
	unsigned m1Rows = m1.rows(), m1Cols = m1.cols(), m2Rows = m2.rows(), m2Cols = m2.cols();
	Matrix result(m1.rows() * m2.rows(), m2.cols() * m2.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m1.get(i / m1Rows, j / m1Cols) * m2.get(i % m2Rows, j % m2Cols);
		}
	}
	return result;
}

Matrix operator*(Matrix const &m1, Matrix const &m2){
	Matrix result(m1.rows(), m1.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m1.get(i, j) * m2.get(i, j);
		}
	}
	return result;
}

Matrix operator*(const double num, Matrix const &m){
	Matrix result(m.rows(), m.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = num * m.get(i, j);
		}
	}
	return result;
}

Matrix operator/(Matrix const &m1, Matrix const &m2){
	Matrix result(m1.rows(), m1.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m1.get(i, j) / m2.get(i, j);
		}
	}
	return result;
}

Matrix operator/(Matrix const &m, double const num){
	Matrix result(m.rows(), m.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m.get(i, j) / num;
		}
	}
	return result;
}

Matrix operator+(Matrix const &m1, Matrix const &m2){
	Matrix result(m1.rows(), m1.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m1.get(i, j) + m2.get(i, j);
		}
	}
	return result;
}

Matrix operator-(Matrix const &m1, Matrix const &m2){
	Matrix result(m1.rows(), m1.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m1.get(i, j) - m2.get(i, j);
		}
	}
	return result;
}

Matrix operator-(const double num, Matrix const &m){
	Matrix result(m.rows(), m.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = num - m.get(i, j);
		}
	}
	return result;
}

Matrix operator~(Matrix const &m){
	Matrix result(m.cols(), m.rows());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = m.get(j, i);
		}
	}
	return result;
}

double sum (Matrix const &m){
	double sum = 0;
	for(unsigned i = 0; i < m.rows(); ++i){
		for(unsigned j = 0; j < m.cols(); ++j){
			sum += m.get(i, j);
		}
	}
	return sum;
}

Matrix matrixFunction(Matrix const &m, double (*function)(double num)){
	Matrix result(m.rows(), m.cols());
	for(unsigned i = 0; i < result.rows(); ++i){
		for(unsigned j = 0; j < result.cols(); ++j){
			result.get(i, j) = function(m.get(i, j));
		}
	}
	return result;
}

Matrix link(int count, ...){
	va_list args;
	va_start(args, count);

	std::vector<std::vector<double> > resultV;
	for(int i = 0; i < count; ++i){
		Matrix temp = va_arg(args, Matrix);
		resultV.insert(resultV.end(), temp.m.begin(), temp.m.end());
	}
	va_end(args);

	Matrix resultM(resultV);
	return resultM;
}
