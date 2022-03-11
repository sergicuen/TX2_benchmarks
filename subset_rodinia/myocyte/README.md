// The following are the command parameters to the application:
// 1) Simulation time interval which is the number of miliseconds to simulate. Needs to be integer > 0
// 2) Number of instances of simulation to run. Needs to be integer > 0.
// 3) Method of parallelization. Need to be 0 for parallelization inside each simulation instance, or 1 for parallelization across instances.
// Example:
// myocyte.out 100 100 1
//
// for more information see main.cu

# \#\#\#\#\#\#\#\#\#\# READ BEFORE USE \#\#\#\#\#\#\#\#\#\#\#\#
At the beginning of this file, there is the original readme text which includes an explanation of the arguments passed to the application.
## Compilation and versions

If the cuda installation is the default one, execute the following command to compile all versions:
`./compile_all.sh`
Some warnings may appear during the compilation, but at the end, ten different binaries inside the **bin** directory should appear.
If you find any problems check the Makefile definitions of the tools.
## Versions
The generated versions are the following:

- **myocyte.out** : Default version. No kernel redundancy
- **myocyte.out_redundant** : Includes the dual kernel redundancy.

The rest of the versions include the following suffixes:

- **_triple** : Triple kernel redundancy
- **_serialize** : Forces the redundant kernel to serialize
- **_timing** : Used to measure specific parts of the execution
The default option to use for simple dual kernel redundancy is the **myocyte.out_redundancy** 
## Arguments and inputs
The default arguments which I have used are the following
100 1 1

Regarding input files, the input files are scripted in the code in work_2.cu file
The required files are **y.txt** and **params.txt** 
Both must be inside a directory named "data" in the same directory where the binary is executed

## Profiler
An example of how to launch it with the profiler:
`/usr/local/cuda/bin/nvprof --print-gpu-trace --print-api-trace -o tracefile.nvvp --csv ./bin/myocyte.out_redundant 100 1 1  &> outputfile.txt`
This execution will generate two files, the **tracefile.nvvp**, a trace that can be opened using the nvvp (Nvidia Visual Profiler), and the 
outputfile.txt. The output file contains both the output of the application, which in the case of the redundant versions includes whether the redundant results match, and a trace in csv (comma separated value) style. 

In some machines, the profiler needs to be executed with sudo permission