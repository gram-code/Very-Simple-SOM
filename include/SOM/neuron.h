#pragma once
#include <vector>

//класс Нейрон
class Neuron
{
public:
    //Конструктор: размерность входных данных
    Neuron(int input_dim);
    
    //Конструктор копий
    Neuron& operator=(const Neuron& neuron);
    
    //Backward: входной вектор, значение функции окрестности, текущая итерация, всего итераций, коэф. для параметра обучения
    void backward(const std::vector<double>& v, const double h, const int iteration, const int iterations, const double eta0);
    
    //Операторы доступа к вектору весов
    double& operator[](const int index);
    double  operator[](const int index) const;
    
private:
    //Размерность входных данных
    const int m_input_dim;
    
    //Вектор весов
    std::vector<double> m_weights;
};
