
#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>
#include<iostream>
#include<fstream>

namespace brian {

std::vector< rk_state* > _mersenne_twister_states;

//////////////// networks /////////////////
Network magicnetwork;

//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 17;
double * _array_neurongroup_I;
const int _num__array_neurongroup_I = 17;
double * _array_neurongroup_n;
const int _num__array_neurongroup_n = 17;
double * _array_neurongroup_V;
const int _num__array_neurongroup_V = 17;
int32_t * _array_statemonitor__indices;
const int _num__array_statemonitor__indices = 17;
int32_t * _array_statemonitor_N;
const int _num__array_statemonitor_N = 1;
double * _array_statemonitor_n;
const int _num__array_statemonitor_n = (0, 17);
double * _array_statemonitor_V;
const int _num__array_statemonitor_V = (0, 17);

//////////////// dynamic arrays 1d /////////
std::vector<double> _dynamic_array_statemonitor_t;

//////////////// dynamic arrays 2d /////////
DynamicArray2D<double> _dynamic_array_statemonitor_n;
DynamicArray2D<double> _dynamic_array_statemonitor_V;

/////////////// static arrays /////////////
double * _static_array__array_neurongroup_I;
const int _num__static_array__array_neurongroup_I = 17;
int32_t * _static_array__array_statemonitor__indices;
const int _num__static_array__array_statemonitor__indices = 17;

//////////////// synapses /////////////////

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
	using namespace brian;

    // Arrays initialized to 0
	_array_defaultclock_dt = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

	_array_defaultclock_t = new double[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

	_array_defaultclock_timestep = new int64_t[1];
    
	for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

	_array_neurongroup_i = new int32_t[17];
    
	for(int i=0; i<17; i++) _array_neurongroup_i[i] = 0;

	_array_neurongroup_I = new double[17];
    
	for(int i=0; i<17; i++) _array_neurongroup_I[i] = 0;

	_array_neurongroup_n = new double[17];
    
	for(int i=0; i<17; i++) _array_neurongroup_n[i] = 0;

	_array_neurongroup_V = new double[17];
    
	for(int i=0; i<17; i++) _array_neurongroup_V[i] = 0;

	_array_statemonitor__indices = new int32_t[17];
    
	for(int i=0; i<17; i++) _array_statemonitor__indices[i] = 0;

	_array_statemonitor_N = new int32_t[1];
    
	for(int i=0; i<1; i++) _array_statemonitor_N[i] = 0;


	// Arrays initialized to an "arange"
	_array_neurongroup_i = new int32_t[17];
    
	for(int i=0; i<17; i++) _array_neurongroup_i[i] = 0 + i;


	// static arrays
	_static_array__array_neurongroup_I = new double[17];
	_static_array__array_statemonitor__indices = new int32_t[17];

	// Random number generator states
	for (int i=0; i<1; i++)
	    _mersenne_twister_states.push_back(new rk_state());
}

void _load_arrays()
{
	using namespace brian;

	ifstream f_static_array__array_neurongroup_I;
	f_static_array__array_neurongroup_I.open("static_arrays/_static_array__array_neurongroup_I", ios::in | ios::binary);
	if(f_static_array__array_neurongroup_I.is_open())
	{
		f_static_array__array_neurongroup_I.read(reinterpret_cast<char*>(_static_array__array_neurongroup_I), 17*sizeof(double));
	} else
	{
		std::cout << "Error opening static array _static_array__array_neurongroup_I." << endl;
	}
	ifstream f_static_array__array_statemonitor__indices;
	f_static_array__array_statemonitor__indices.open("static_arrays/_static_array__array_statemonitor__indices", ios::in | ios::binary);
	if(f_static_array__array_statemonitor__indices.is_open())
	{
		f_static_array__array_statemonitor__indices.read(reinterpret_cast<char*>(_static_array__array_statemonitor__indices), 17*sizeof(int32_t));
	} else
	{
		std::cout << "Error opening static array _static_array__array_statemonitor__indices." << endl;
	}
}

void _write_arrays()
{
	using namespace brian;

	ofstream outfile__array_defaultclock_dt;
	outfile__array_defaultclock_dt.open("results/_array_defaultclock_dt_1978099143", ios::binary | ios::out);
	if(outfile__array_defaultclock_dt.is_open())
	{
		outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
		outfile__array_defaultclock_dt.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
	}
	ofstream outfile__array_defaultclock_t;
	outfile__array_defaultclock_t.open("results/_array_defaultclock_t_2669362164", ios::binary | ios::out);
	if(outfile__array_defaultclock_t.is_open())
	{
		outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
		outfile__array_defaultclock_t.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_t." << endl;
	}
	ofstream outfile__array_defaultclock_timestep;
	outfile__array_defaultclock_timestep.open("results/_array_defaultclock_timestep_144223508", ios::binary | ios::out);
	if(outfile__array_defaultclock_timestep.is_open())
	{
		outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
		outfile__array_defaultclock_timestep.close();
	} else
	{
		std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
	}
	ofstream outfile__array_neurongroup_i;
	outfile__array_neurongroup_i.open("results/_array_neurongroup_i_2649026944", ios::binary | ios::out);
	if(outfile__array_neurongroup_i.is_open())
	{
		outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 17*sizeof(_array_neurongroup_i[0]));
		outfile__array_neurongroup_i.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_i." << endl;
	}
	ofstream outfile__array_neurongroup_I;
	outfile__array_neurongroup_I.open("results/_array_neurongroup_I_2794115400", ios::binary | ios::out);
	if(outfile__array_neurongroup_I.is_open())
	{
		outfile__array_neurongroup_I.write(reinterpret_cast<char*>(_array_neurongroup_I), 17*sizeof(_array_neurongroup_I[0]));
		outfile__array_neurongroup_I.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_I." << endl;
	}
	ofstream outfile__array_neurongroup_n;
	outfile__array_neurongroup_n.open("results/_array_neurongroup_n_58745891", ios::binary | ios::out);
	if(outfile__array_neurongroup_n.is_open())
	{
		outfile__array_neurongroup_n.write(reinterpret_cast<char*>(_array_neurongroup_n), 17*sizeof(_array_neurongroup_n[0]));
		outfile__array_neurongroup_n.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_n." << endl;
	}
	ofstream outfile__array_neurongroup_V;
	outfile__array_neurongroup_V.open("results/_array_neurongroup_V_729996477", ios::binary | ios::out);
	if(outfile__array_neurongroup_V.is_open())
	{
		outfile__array_neurongroup_V.write(reinterpret_cast<char*>(_array_neurongroup_V), 17*sizeof(_array_neurongroup_V[0]));
		outfile__array_neurongroup_V.close();
	} else
	{
		std::cout << "Error writing output file for _array_neurongroup_V." << endl;
	}
	ofstream outfile__array_statemonitor__indices;
	outfile__array_statemonitor__indices.open("results/_array_statemonitor__indices_2854283999", ios::binary | ios::out);
	if(outfile__array_statemonitor__indices.is_open())
	{
		outfile__array_statemonitor__indices.write(reinterpret_cast<char*>(_array_statemonitor__indices), 17*sizeof(_array_statemonitor__indices[0]));
		outfile__array_statemonitor__indices.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor__indices." << endl;
	}
	ofstream outfile__array_statemonitor_N;
	outfile__array_statemonitor_N.open("results/_array_statemonitor_N_4140778434", ios::binary | ios::out);
	if(outfile__array_statemonitor_N.is_open())
	{
		outfile__array_statemonitor_N.write(reinterpret_cast<char*>(_array_statemonitor_N), 1*sizeof(_array_statemonitor_N[0]));
		outfile__array_statemonitor_N.close();
	} else
	{
		std::cout << "Error writing output file for _array_statemonitor_N." << endl;
	}

	ofstream outfile__dynamic_array_statemonitor_t;
	outfile__dynamic_array_statemonitor_t.open("results/_dynamic_array_statemonitor_t_3983503110", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_t.is_open())
	{
        if (! _dynamic_array_statemonitor_t.empty() )
        {
			outfile__dynamic_array_statemonitor_t.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_t[0]), _dynamic_array_statemonitor_t.size()*sizeof(_dynamic_array_statemonitor_t[0]));
		    outfile__dynamic_array_statemonitor_t.close();
		}
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_t." << endl;
	}

	ofstream outfile__dynamic_array_statemonitor_n;
	outfile__dynamic_array_statemonitor_n.open("results/_dynamic_array_statemonitor_n_269325948", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_n.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_n.n; n++)
        {
            if (! _dynamic_array_statemonitor_n(n).empty())
            {
                outfile__dynamic_array_statemonitor_n.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_n(n, 0)), _dynamic_array_statemonitor_n.m*sizeof(_dynamic_array_statemonitor_n(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_n.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_n." << endl;
	}
	ofstream outfile__dynamic_array_statemonitor_V;
	outfile__dynamic_array_statemonitor_V.open("results/_dynamic_array_statemonitor_V_940519138", ios::binary | ios::out);
	if(outfile__dynamic_array_statemonitor_V.is_open())
	{
        for (int n=0; n<_dynamic_array_statemonitor_V.n; n++)
        {
            if (! _dynamic_array_statemonitor_V(n).empty())
            {
                outfile__dynamic_array_statemonitor_V.write(reinterpret_cast<char*>(&_dynamic_array_statemonitor_V(n, 0)), _dynamic_array_statemonitor_V.m*sizeof(_dynamic_array_statemonitor_V(0, 0)));
            }
        }
        outfile__dynamic_array_statemonitor_V.close();
	} else
	{
		std::cout << "Error writing output file for _dynamic_array_statemonitor_V." << endl;
	}
	// Write last run info to disk
	ofstream outfile_last_run_info;
	outfile_last_run_info.open("results/last_run_info.txt", ios::out);
	if(outfile_last_run_info.is_open())
	{
		outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
		outfile_last_run_info.close();
	} else
	{
	    std::cout << "Error writing last run info to file." << std::endl;
	}
}

void _dealloc_arrays()
{
	using namespace brian;


	// static arrays
	if(_static_array__array_neurongroup_I!=0)
	{
		delete [] _static_array__array_neurongroup_I;
		_static_array__array_neurongroup_I = 0;
	}
	if(_static_array__array_statemonitor__indices!=0)
	{
		delete [] _static_array__array_statemonitor__indices;
		_static_array__array_statemonitor__indices = 0;
	}
}

