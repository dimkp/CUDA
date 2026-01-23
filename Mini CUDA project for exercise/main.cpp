#include <iostream>
#include <cstring>
#include "KernelHeader.cuh"

void printMat(std::string msg, const int* arr, const int size)
{
	printf("======================\n");
	printf("%s\n", msg.c_str());
	for (int i = 0; i < size; i++)
		printf("%d\n", arr[i]);
	printf("======================\n");
}


int main()
{	
	int* x;
	int* y;
	int size = 100;
	int byteSize = size * sizeof(int);
	x = (int*)malloc(byteSize);
	y = (int*)malloc(byteSize);

	// Array copy
	if (!x || !y)
	{
		free(x);
		free(y);
		printf("Memory failure: host\n");
		exit(1);
	}
	for (int i = 0; i < size; i++)
		x[i] = i + 1;
	
	CopyKernelCall(x ,y ,size, byteSize); 
	//printMat("Array copy", y, size);

	// Array addition 
	int* z;
	z = (int*)malloc(byteSize);
	if (!z)
	{
		free(x);
		free(y);
		free(z);
		printf("Memory failure: host\n");
		exit(1);
	}
	AddKernelCall(x, y, z, size, byteSize);
	//printMat("Array addition", z, size);

	// Dot product of 2 arrays
	int* result = (int*)malloc(sizeof(int));
	if (!result)
	{
		free(x);
		free(y);
		free(z);
		free(result);
		printf("Memory failure: host\n");
		exit(1);
	}
	DotProductKernelCall(x, y, result, size, byteSize);
	printf("======================\n");
	printf("GPU Result of Dot Product: %d\n", *result);
	printf("======================\n");



	// Memory cleanup
	free(x);
	free(y);
	free(z);
	free(result);
	return 0;
}
