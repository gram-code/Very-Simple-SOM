#include "../include/SOM/neuron.h"
#include <vector>
#include <random>
#include <cmath>

//Конструктор: размерность входных данных
Neuron::Neuron(int input_dim): m_input_dim(input_dim)
{
    //Инициализация вектора весов случайными значениями
    m_weights.resize(m_input_dim);
    
    //Вихрь Месенна
    std::mt19937_64 seed(std::random_device{}());
    std::uniform_real_distribution<double> random_for_weights(0.0, 1.0);
    
    for(int i = 0; i < m_input_dim; i++)
    {   
        m_weights[i] = random_for_weights(seed);
    }
    
}

//Конструктор копий
Neuron& Neuron::operator=(const Neuron& neuron)
{
    for(int i = 0; i < m_input_dim; i++)
        m_weights[i] = neuron[i];
    return *this;  
}

//Backward: входной вектор, значение функции окрестности, текущая итерация, всего итераций, коэф. для параметра обучения
void Neuron::backward(const std::vector<double>& v, const double h, const int iteration, const int iterations, const double eta0)
{
    //Параметр для предотвращения метастабильного состояния
    const double eta = eta0 * std::exp(- (iteration/iterations) );
    for(int i = 0; i < m_input_dim; i++)
        m_weights[i] = m_weights[i] + eta * h * (v[i] - m_weights[i]);
}


//Операторы доступа к вектору весов
double& Neuron::Neuron::operator[](const int index) 
{
    return m_weights[index];
    
}

double Neuron::operator[](const int index) const 
{
    return m_weights[index];
}
