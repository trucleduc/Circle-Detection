Code for the paper:
Truc Le, Ye Duan, "Circle Detection on Images by Line Segment and 
Circle Completeness", in the IEEE International Conference on 
Image Processing, 2016.

---------- COMPLILATION ----------
1. The LSD code must be compiled first. See the following: 
"LSD: a Line Segment Detector" by Rafael Grompone von Gioi,
    Jeremie Jakubowicz, Jean-Michel Morel, and Gregory Randall,
    Image Processing On Line, 2012. DOI:10.5201/ipol.2012.gjmr-lsd
    http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd

2. Compile all the C++ files in the "core" directory.
	mex clusterByDistance.cpp
	mex meanShift.cpp
	Unix: mex -v -largeArrayDims estimateNormals.cpp -lmwlapack
	Windows: mex('-v', '-largeArrayDims', 'estimateNormals.cpp', fullfile(matlabroot, 'extern', 'lib', computer('arch'), 'microsoft', 'libmwlapack.lib'));

---------- RUN ----------
"demo.m" file shows how to run the code.

If you use the code for your research, please cite our paper:

Truc Le, Ye Duan, "Circle Detection on Images by Line Segment and 
Circle Completeness", in the IEEE International Conference on 
Image Processing, 2016.
