#pragma once 
#include "../include/SOM/neuron.h"
#include <vector>
#include <string>
#include <utility>

//Самоорганизующаяся карта Кохонена
class Network
{
public:
    //Конструктор: набор данных, размерность входных данных, размерность выходных данных,количество итераций, коэффициент в функции изменения весов, коэф. в функции окрестности 
    Network(const std::vector<std::vector<double>>& data, int input_dim, int output_dim, int iterations, double eta, double o0);
    
    //Функция обучения: выводить каждую print_n итерацию цикла
    void train(const int print_n);
    
    //Вывод сети в файл: имя файла
    void print(std::string filename) const;
    
private:
    //---Подготовка данных---
    
    //Поиск максимума среди признака
    double max_of_input(int i) const;
    
    //Поиск максимума среди признака
    double min_of_input(int i) const;
    
    //Нормализация входных данных
    void normalization();
    
    //---Процесс конкуреции---
    
    //Метрика Минковского: Нейрон, входной вектор, лямбда (по умолчанию 2)
    double metrics(const Neuron& n, const std::vector<double>& v, const int lambda) const;
    
    //Поиск минимальной дистанции: входной вектор
    std::pair<int,int> minimal_distance(const std::vector<double>& v) const;
    
    
    //---Процесс кооперации и активации---
    
    //Функция окрестности: номер текущей итерации, коорд. нейрона победителя, коорд. нейрона
    double neighborhood(const int iteration, const int winner_i, const int winner_j, const int i, const int j) const;
    
    //Поиск нейронов, которые попадают под радиус нейрона-победителя и их активация: коорд, нейрона победителя, входной вектор.
    void cooperation_and_adaptation(const int iteration, const std::pair<int, int>& winner, const std::vector<double>& v);
    
    
    //---Переменные---
    
    //Набор входных векторов
    std::vector<std::vector<double>> m_input; 
    
    //Двумерный слой нейронов
    std::vector<std::vector<Neuron>> m_output;  
    
    //Размерность входных данных
    const int m_input_dim;  
    
    //Размер двумерного слоя нейронов. 
    const int m_output_dim;
    
    //Количество итераций обучения
    const int m_iterations;
    
    //Коэффицент для функции изменения весов нейрона (см. cooperation_and_adaptation)
    const double m_eta0;
    
    //Коэффициент для Гауссовой функции (функции neighborhood)
    const double m_o0;
};
