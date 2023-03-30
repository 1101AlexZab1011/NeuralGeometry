#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "objects.h"
#include "brianlib/common_math.h"
#include "brianlib/stdint_compat.h"
#include<cmath>
#include<ctime>
#include<iostream>
#include<fstream>
#include<climits>

////// SUPPORT CODE ///////
namespace {
        
    template < typename T1, typename T2 > struct _higher_type;
    template < > struct _higher_type<int,int> { typedef int type; };
    template < > struct _higher_type<int,long> { typedef long type; };
    template < > struct _higher_type<int,long long> { typedef long long type; };
    template < > struct _higher_type<int,float> { typedef float type; };
    template < > struct _higher_type<int,double> { typedef double type; };
    template < > struct _higher_type<int,long double> { typedef long double type; };
    template < > struct _higher_type<long,int> { typedef long type; };
    template < > struct _higher_type<long,long> { typedef long type; };
    template < > struct _higher_type<long,long long> { typedef long long type; };
    template < > struct _higher_type<long,float> { typedef float type; };
    template < > struct _higher_type<long,double> { typedef double type; };
    template < > struct _higher_type<long,long double> { typedef long double type; };
    template < > struct _higher_type<long long,int> { typedef long long type; };
    template < > struct _higher_type<long long,long> { typedef long long type; };
    template < > struct _higher_type<long long,long long> { typedef long long type; };
    template < > struct _higher_type<long long,float> { typedef float type; };
    template < > struct _higher_type<long long,double> { typedef double type; };
    template < > struct _higher_type<long long,long double> { typedef long double type; };
    template < > struct _higher_type<float,int> { typedef float type; };
    template < > struct _higher_type<float,long> { typedef float type; };
    template < > struct _higher_type<float,long long> { typedef float type; };
    template < > struct _higher_type<float,float> { typedef float type; };
    template < > struct _higher_type<float,double> { typedef double type; };
    template < > struct _higher_type<float,long double> { typedef long double type; };
    template < > struct _higher_type<double,int> { typedef double type; };
    template < > struct _higher_type<double,long> { typedef double type; };
    template < > struct _higher_type<double,long long> { typedef double type; };
    template < > struct _higher_type<double,float> { typedef double type; };
    template < > struct _higher_type<double,double> { typedef double type; };
    template < > struct _higher_type<double,long double> { typedef long double type; };
    template < > struct _higher_type<long double,int> { typedef long double type; };
    template < > struct _higher_type<long double,long> { typedef long double type; };
    template < > struct _higher_type<long double,long long> { typedef long double type; };
    template < > struct _higher_type<long double,float> { typedef long double type; };
    template < > struct _higher_type<long double,double> { typedef long double type; };
    template < > struct _higher_type<long double,long double> { typedef long double type; };
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_mod(T1 x, T2 y)
    {{
        return x-y*floor(1.0*x/y);
    }}
    template < typename T1, typename T2 >
    static inline typename _higher_type<T1,T2>::type
    _brian_floordiv(T1 x, T2 y)
    {{
        return floor(1.0*x/y);
    }}
    #ifdef _MSC_VER
    #define _brian_pow(x, y) (pow((double)(x), (y)))
    #else
    #define _brian_pow(x, y) (pow((x), (y)))
    #endif

}

////// HASH DEFINES ///////



void _run_neurongroup_stateupdater_codeobject()
{
    using namespace brian;


    ///// CONSTANTS ///////////
    const double C = 1e-06;
const size_t _numI = 17;
const int64_t N = 17;
const size_t _numV = 17;
const double V_1 = 0.01;
const double V_2 = 0.015;
const double V_3 = - 0.001;
const double V_4 = 0.0145;
const double V_Ca = 0.1;
const double V_K = - 0.07;
const double V_L = - 0.05;
const size_t _numdt = 1;
const double g_Ca = 0.001;
const double g_K = 0.002;
const double g_L = 0.0005;
const double lambda_n__max = 66.66666666666667;
const size_t _numn = 17;
    ///// POINTERS ////////////
        
    double* __restrict  _ptr_array_neurongroup_I = _array_neurongroup_I;
    double* __restrict  _ptr_array_neurongroup_V = _array_neurongroup_V;
    double*   _ptr_array_defaultclock_dt = _array_defaultclock_dt;
    double* __restrict  _ptr_array_neurongroup_n = _array_neurongroup_n;


    //// MAIN CODE ////////////
    // scalar code
    const size_t _vectorisation_idx = -1;
        
    const double dt = _ptr_array_defaultclock_dt[0];
    const double _lio_1 = (1.0f*(0.5 * (V_Ca * g_Ca))/C) + (1.0f*(V_L * g_L)/C);
    const double _lio_2 = 1.0f*1.0/C;
    const double _lio_3 = 1.0f*(0.5 * (V_Ca * g_Ca))/C;
    const double _lio_4 = 0.0 - (1.0f*V_1/V_2);
    const double _lio_5 = 1.0f*1.0/V_2;
    const double _lio_6 = 1.0f*(V_K * g_K)/C;
    const double _lio_7 = 0.0 - ((1.0f*(0.5 * g_Ca)/C) + (1.0f*g_L/C));
    const double _lio_8 = 1.0f*((- 0.5) * g_Ca)/C;
    const double _lio_9 = 1.0f*g_K/C;
    const double _lio_10 = 1.0f*1.0/lambda_n__max;
    const double _lio_11 = 0.5 * lambda_n__max;
    const double _lio_12 = 0.0 - (1.0f*(0.5 * V_3)/V_4);
    const double _lio_13 = 1.0f*0.5/V_4;
    const double _lio_14 = 0.0 - (1.0f*V_3/V_4);
    const double _lio_15 = 1.0f*1.0/V_4;
    const double _lio_16 = (- dt) * lambda_n__max;


    const int _N = N;
    
    for(int _idx=0; _idx<_N; _idx++)
    {
        // vector code
        const size_t _vectorisation_idx = _idx;
                
        const double I = _ptr_array_neurongroup_I[_idx];
        double V = _ptr_array_neurongroup_V[_idx];
        double n = _ptr_array_neurongroup_n[_idx];
        const double _BA_V = 1.0f*(_lio_1 + (((_lio_2 * I) + (_lio_3 * tanh(_lio_4 + (_lio_5 * V)))) + (_lio_6 * n)))/((_lio_7 + (_lio_8 * tanh(_lio_4 + (_lio_5 * V)))) - (_lio_9 * n));
        const double _V = (- _BA_V) + ((V + _BA_V) * exp(dt * ((_lio_7 + (_lio_8 * tanh(_lio_4 + (_lio_5 * V)))) - (_lio_9 * n))));
        const double _BA_n = 1.0f*(_lio_10 * (- ((_lio_11 * (cosh(_lio_12 + (_lio_13 * V)) * tanh(_lio_14 + (_lio_15 * V)))) + (_lio_11 * cosh(_lio_12 + (_lio_13 * V))))))/cosh(_lio_12 + (_lio_13 * V));
        const double _n = (- _BA_n) + ((_BA_n + n) * exp(_lio_16 * cosh(_lio_12 + (_lio_13 * V))));
        V = _V;
        n = _n;
        _ptr_array_neurongroup_V[_idx] = V;
        _ptr_array_neurongroup_n[_idx] = n;

    }

}


