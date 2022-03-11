Must have the CUDA Toolkit installed and nvcc working
To build and run nearest neighbor:
	make nn
	./nn filelist_4 -r 3 -lat 30 -lng 90

To generate new data sets:
	Edit gen_dataset.sh and select the size of the desired data set
	make hurricane_gen
	./hurricane_gen <num records> <num files>

Full Usage:

  nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-h]
  
  example:
  $ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90
  
  filename     the filename that lists the data input files
  -r [int]     the number of records to return (default: 10)
  -lat [float] the latitude for nearest neighbors (default: 0)
  -lng [float] the longitude for nearest neighbors (default: 0)
  
  -h, --help   Display the help file  
  
  Note: The filename is required as the first parameter.

# \#\#\#\#\#\#\#\#\#\# READ BEFORE USE \#\#\#\#\#\#\#\#\#\#\#\#
At the beginning of this file, there is the original readme text which includes an explanation of the arguments passed to the application.
## Compilation and versions

If the cuda installation is the default one, execute the following command to compile all versions:
`./compile_all.sh`
Some warnings may appear during the compilation, but at the end, ten different binaries inside the **bin** directory should appear.
If you find any problems check the Makefile definitions of the tools.
## Versions
The generated versions are the following:

- **nn** : Default version. No kernel redundancy
- **nn_redundant** : Includes the dual kernel redundancy.

The rest of the versions include the following suffixes:

- **_triple** : Triple kernel redundancy
- **_serialize** : Forces the redundant kernel to serialize
- **_timing** : Used to measure specific parts of the execution

The default option to use for simple dual kernel redundancy is the **nn_redundancy**.
## Arguments and inputs
Because of the modifications we made in the code to allow the kernels to be friendly, more arguments than the default application are required:
Original arguments include (explained on the top of this file):

- filename : File which lists the path of the different input files (with extension .db) that will be used as database
-r [int] : The number of records to return (default: 10)
-lat [float] : The latitude for nearest neighbors (default: 0)
-lng [float] : The longitude for nearest neighbors (default: 0)
- ..., others I do not use and have default values already, such as : q [int] Quiet?, disable prints

Please ensure that filelist_64 correctly points to the database files. I would reccommend to check the contents of this file first to be sure.
If an error message appears on the executions mentioning the an error while reading the db files, probably filelist_64 would be the source of the problem.

### New arguments added
If you read the mechanism we presented on the Journal, two different factors are required to shape the kernel. One is named as **Thread Coarsening** and the other one is **Block Division**
- w [int] : Thread Coarsening for the first kernel
- f [int] : Block Divison for the first kernel
- s [int] : Thread Coarsening for the second kernel
- a [int] : Block Divison for the second kernel

If these extra parameters (w,f,s,a) are all set to 1, no modifications are made to the default kernel shape.
## Heavy and Friendly executions
Heavy and friendly execution depends not only on the software but also is platform dependent as well. The following options have been tested in a 
Tegra TX2 we have in our lab. Therefore, should be reproducible in you Tegra as well.
The specification of the used GPU can be seen in the file **device_quert_output.txt** which is the output of the device query command and should be very similar to yours. 

### Heavy
In order to have a heavy execution, no modifications are required in the original kernel shape, the binary then will be executed like this:
`bin/nn_redundant filelist_64 -r 1000 -lat 30 -q 0 -w 1 -f 1 -s 1 -a 1`
I've let you a file named as default_arguments with the default (heavy kernel) arguments

### Friendly
Instead, modifications are required to execute the kernels as friendly
`bin/nn_redundant filelist_64 -r 1000 -lat 30 -q 0 -w 850 -f 1 -s 850 -a 1`
I've let you a file named as concurrent_arguments with the modified (friendly kernel) arguments


## Profiler
An example of how to launch it with the profiler:
`/usr/local/cuda/bin/nvprof --print-gpu-trace --print-api-trace -o tracefile.nvvp --csv bin/nn_redundant filelist_64 -r 1000 -lat 30 -q 0 -w 850 -f 1 -s 850 -a 1 &> outputfile`
This execution will generate two files, the **tracefile.nvvp**, a trace that can be opened using the nvvp (Nvidia Visual Profiler) and the 
outputfile.txt. The output file contains both the output of the application, which in the case of the redundant versions includes whether the redundant results match, and a trace in csv (comma separated value) style. 

In some machines, the profiler needs to be executed with sudo permission