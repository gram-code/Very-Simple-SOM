#include "../include/SOM/network.h"
#include <vector>
#include <string>
#include <utility>
#include <cmath>

#include <iostream>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


//Конструктор: набор данных, размерность входных данных, размерность выходных данных,количество итераций, коэффициент в функции изменения весов, коэф. в функции окрестности 
Network::Network(const std::vector<std::vector<double>>& data, int input_dim, int output_dim, int iterations, double eta, double o0): 
m_input_dim(input_dim), m_output_dim(output_dim), m_iterations(iterations), m_eta0(eta), m_o0(o0)
{
    //Иницилиазация нейронов
    m_output.resize(m_output_dim, std::vector<Neuron>());
    
    for(int i = 0; i < m_output_dim; i++)
        for(int j = 0; j < m_output_dim; j++)
            m_output[i].push_back(Neuron(m_input_dim));
    
    //Инициализация входных векторов
    m_input.resize(data.size(), std::vector<double>(m_input_dim, 0));
    
    for(int i = 0; i < m_input.size(); i++)
    {
        for(int j = 0; j < m_input_dim; j++)
        {
            m_input[i][j] = data[i][j];
        }
        
    }
    //Нормализцаия входных данных
    normalization();
}

//---Интерфейс класса---

//Функция обучения: выводить каждую print_n итерацию цикла
void Network::train(const int print_n)
{
    //Номер входного вектора из множества входных данных
    
    int i = 0;
    
    //Количество итераций
    for( int iteration = 1; iteration < m_iterations; iteration++)
    {
        //Такой способ используется вместо вложенного цикла, чтобы число итераций соответсвовало действительности, а не было в  m_input.size() раз больше.
        i %= m_input.size(); 
        
        //Процесс конкуренции
        std::pair<int,int> winner = minimal_distance(m_input[i]);
        //std::cout << winner.first << " " << winner.second << "\n";
        
        //Процесс кооперации и адаптации
        cooperation_and_adaptation(iteration, winner, m_input[i]);
        
        //Вывод каждой print_n итерации цикла
        if( (print_n && !(iteration % print_n) ) )
        {
            print(std::to_string(iteration) + ".jpg" );
        }
        
        i++;
    }
}

//Вывод сети в файл: имя файла
void Network::print(std::string filename) const
{
    cv::Mat image = cv::Mat_<cv::Vec3b>(m_output_dim, m_output_dim);
    cv::Mat image2 = cv::Mat_<cv::Vec3b>(m_output_dim, m_output_dim);
    cv::Mat image3 = cv::Mat_<cv::Vec3b>(m_output_dim, m_output_dim);
    cv::Mat image4 = cv::Mat_<cv::Vec3b>(m_output_dim, m_output_dim);
    cv::Mat image5 = cv::Mat_<cv::Vec3b>(m_output_dim, m_output_dim);
    
    for(int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++) 
        {            
            cv::Vec3b& pixel = image.at<cv::Vec3b>(cv::Point(x, y));
            //BGR
            pixel[0] = m_output[x][y][0] * 255;
            pixel[1] = m_output[x][y][0] * 255;
            pixel[2] = m_output[x][y][0] * 255;
            
            cv::Vec3b& pixel2 = image2.at<cv::Vec3b>(cv::Point(x, y));
            //BGR
            pixel2[0] = m_output[x][y][1] * 255;
            pixel2[1] = m_output[x][y][1] * 255;
            pixel2[2] = m_output[x][y][1] * 255;
            
            cv::Vec3b& pixel3 = image3.at<cv::Vec3b>(cv::Point(x, y));
            //BGR
            pixel3[0] = m_output[x][y][2] * 255;
            pixel3[1] = m_output[x][y][2] * 255;
            pixel3[2] = m_output[x][y][2] * 255;
            
            cv::Vec3b& pixel4 = image4.at<cv::Vec3b>(cv::Point(x, y));
            //BGR
            pixel4[0] = m_output[x][y][3] * 255;
            pixel4[1] = m_output[x][y][3] * 255;
            pixel4[2] = m_output[x][y][3] * 255;
            
            cv::Vec3b& pixel5 = image5.at<cv::Vec3b>(cv::Point(x, y));
            //BGR
            pixel5[2] = (1 - m_output[x][y][0])*(1 - m_output[x][y][3]) * 255;
            pixel5[1] = (1 - m_output[x][y][1])*(1 - m_output[x][y][3]) * 255;
            pixel5[0] = (1 - m_output[x][y][1])*(1 - m_output[x][y][3]) * 255;
        }
    }
    //std::cout << image << '\n';
    cv::imwrite("1_"+filename, image);
    cv::imwrite("2_"+filename, image2);
    cv::imwrite("3_"+filename, image3);
    cv::imwrite("4_"+filename, image4);
    cv::imwrite("5_"+filename, image5);
}

//---Нормализация---

//Поиск максимума среди признака
double Network::min_of_input(int i) const
{
    double min;

    for(int j = 0; j < m_input.size(); j++)
    {
        if( !(j) || (m_input[j][i] < min) )
        {
            min = m_input[j][i];
        }
    }
    return min;
}

//Поиск максимума среди признака
double Network::max_of_input(int i) const
{
    double max;
    for(int j = 0; j < m_input.size(); j++)
    {
        if( !(j) || (m_input[j][i] > max) )
        {
            max = m_input[j][i];
        }
        
    }
    return max;
}

//Нормализцаия
void Network::normalization() 
{

    //Нормализация
    for(int i = 0; i < m_input[0].size(); i++)
    {
        //Поиск максимума и мнинмума
        double min = min_of_input(i);
        double max = max_of_input(i);
        for(int j = 0; j < m_input.size(); j++)
        {
            m_input[j][i] = (m_input[j][i] - min) / (max - min);
        }
    }
}

//---Процесс конкуренции---

//Метрика Минковского: Нейрон, входной вектор, лямбда (по умолчанию 2) 
double Network::metrics(const Neuron& n, const std::vector<double>& v, const int lambda = 2) const
{
    double dist = 0.0;
    for(int i = 0; i < m_input_dim; i++)
        dist +=  std::pow( (v[i] - n[i]) , lambda); 
    
    return std::pow(dist, 1.0/lambda);
}

//Поиск минимальной дистанции: входной вектор
std::pair<int,int> Network::minimal_distance(const std::vector<double>& v) const
{        
    double min_dist;
    int min_i, min_j;
    
    for(int i = 0; i < m_output_dim; i++)
    {
        for(int j = 0; j < m_output_dim; j++)
        {
            double dist = metrics(m_output[i][j], v); 
            if(!(i+j) || (dist < min_dist) )
            {
                min_dist = dist;
                min_i = i;
                min_j = j;
            }
        }
    }
    return std::make_pair(min_i, min_j);
}

//---Процесс кооперации и адаптации---

//Функция окрестности: номер текущей итерации, коорд. нейрона победителя, коорд. нейрона
double Network::neighborhood(const int iteration, const int winner_i, const int winner_j, const int i, const int j ) const
{
    //Евклидово расстояние между точками в квадрате
    double d = std::pow(winner_i - i, 2) + std::pow(winner_j - j, 2);
    
    //экспоненциальное убывание, которое зависит от дискретного времени
    double o = m_o0 * std::exp(- ( (iteration*std::log(m_o0)) / m_iterations) ); 
    
    //Гауссова функция 
    return std::exp( -(d/(2*o*o)) ); 
}

//Поиск нейронов, которые попадают под радиус нейрона-победителя и их активация: коорд, нейрона победителя, входной вектор.
void Network::cooperation_and_adaptation(const int iteration, const std::pair<int, int>& winner, const std::vector<double>& v)
{
    for(int i = 0; i < m_output_dim; i++)
    {
        for(int j = 0; j < m_output_dim; j++)
        {
            double h = neighborhood(iteration, winner.first, winner.second, i, j);
            m_output[i][j].backward(v, h, iteration, m_iterations, m_eta0);
        }
    }
}



