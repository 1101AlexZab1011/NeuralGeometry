#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/neurongroup_group_variable_set_conditional_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/statemonitor_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>




int main(int argc, char **argv)
{
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 1e-05;
        
                        
                        for(int i=0; i<_num__array_neurongroup_I; i++)
                        {
                            _array_neurongroup_I[i] = _static_array__array_neurongroup_I[i];
                        }
                        
        
                        
                        for(int i=0; i<_num__array_neurongroup_V; i++)
                        {
                            _array_neurongroup_V[i] = - 0.05;
                        }
                        
        _run_neurongroup_group_variable_set_conditional_codeobject();
        
                        
                        for(int i=0; i<_num__array_statemonitor__indices; i++)
                        {
                            _array_statemonitor__indices[i] = _static_array__array_statemonitor__indices[i];
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _run_statemonitor_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.run(0.22, NULL, 10.0);

	}
        

	brian_end();
        

	return 0;
}