#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <vector>
#include <cstdarg>

class Matrix{
public:
	Matrix();
	Matrix(std::vector<double> const &v);
	Matrix(std::vector<std::vector<double> > const &v);
	Matrix(const unsigned rows, const unsigned cols);
	Matrix(const unsigned rows, const unsigned cols, double const &num);
	Matrix(const unsigned rows, const unsigned cols, double (*initializer)());

	double get(const unsigned row, const unsigned col) const;
	double& get(const unsigned row, const unsigned col);
	Matrix getRow(const unsigned row);
	double set(const unsigned row, const unsigned col, const unsigned val);
	double rows() const;
	double cols() const;

	friend std::ostream& operator<<(std::ostream& os, const Matrix &m);
	friend Matrix link(int count, ...);
private:
	std::vector<std::vector<double> > m;
};
Matrix operator^(Matrix const &m1, Matrix const &m2);
Matrix operator%(Matrix const &m1, Matrix const &m2);

Matrix operator*(Matrix const &m1, Matrix const &m2);
Matrix operator*(const double num, Matrix const &m);

Matrix operator/(Matrix const &m1, Matrix const &m2);
Matrix operator/(Matrix const &m, const double num);

Matrix operator+(Matrix const &m1, Matrix const &m2);

Matrix operator-(Matrix const &m1, Matrix const &m2);
Matrix operator-(const double num, Matrix const &m);

Matrix operator~(Matrix const &m);

double sum(Matrix const &m);
Matrix matrixFunction(Matrix const &m, double (*function)(double num));
#endif