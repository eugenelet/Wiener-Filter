Image Enhancement 
=================

This is an example of the implementation of Wiener Filter in different situations.

The containing executable file is compiled for x86 architecture under Linux.

I've used OpenCV for reading and writing bmp files in this work, so any version of OpenCV will do since they share the same function for reading and writing files. (I'm using version 2.14.3) 

To compile
> **make** 

To run the executable
> **./[executable file name] [input file] [ans file]**

Demo
----
Gaussian Blur
**Input**
![Alt text](input1.bmp?raw=true "Input Image")
**Output**
![Alt text](output1.bmp?raw=true "Output Image")


Motion Blur
**Input**
![Alt text](input2.bmp?raw=true "Input Image")
**Output**
![Alt text](output2.bmp?raw=true "Output Image")


Gaussian Blur + Noise
**Input**
![Alt text](input3.bmp?raw=true "Input Image")
**Output**
![Alt text](output3.bmp?raw=true "Output Image")
