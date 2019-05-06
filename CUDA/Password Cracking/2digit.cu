#include <stdio.h>
#include <cuda_runtime_api.h>

/****************************************************************************
  This program gives an example of a poor way to implement a password cracker
  in CUDA C. It is poor because it acheives this with just one thread, which
  is obviously not good given the scale of parallelism available to CUDA
  programs.
  
  The intentions of this program are:
    1) Demonstrate the use of __device__ and __global__ functions
    2) Enable a simulation of password cracking in the absence of library 
       with equivalent functionality to libcrypt. The password to be found
       is hardcoded into a function called is_a_match.   

  Compile and run with:
    nvcc -o 2digit 2digit.cu

    ./2digit

*** Nirdeshika KC***
*****************************************************************************/


__device__ int is_a_match(char *attempt) {
  char plain_password1[] = "CV78";
  char plain_password2[] = "FT83";
  char plain_password3[] = "HR23";
  char plain_password4[] = "SA32";
  
  char *a = attempt;
  char *b = attempt;
  char *c = attempt;
  char *d = attempt;
  
  char *p1 = plain_password1;
  char *p2 = plain_password2;
  char *p3 = plain_password3;
  char *p4 = plain_password4;
  
  
  while(*a == *p1) {
    if(*a == '\0') {
       printf("Password: %s\n",plain_password1);
    break;
    }
    a++;
    p1++;
  }
  while(*b == *p2) {
    if(*b == '\0') {
      printf("Password: %s\n",plain_password2);
    break;
    }
    b++;
    p2++;
  }
  while(*c == *p3) {
    if(*c == '\0') {
     printf("Password: %s\n",plain_password3);
    break;
    }
    c++;
    p3++;
  }
  while(*d == *p4) {
    if(*d == '\0') {
      printf("Password: %s\n",plain_password4);
    break;
    }
    d++;
    p4++;
  }
  return 0;
}

/****************************************************************************
  The kernel function assume that there will be only one thread and the another one 
  will be block in which the thread is pass one by one to match the password and there are the nested loop 
  to generate the numbers 
*****************************************************************************/

__global__ void  kernel() {
  char a, b;
  int w, y;
  
  char password[5];
  password[4] = '\0'; 
  a = blockIdx.x+65;
  b = threadIdx.x+65;
  
  password[0] =a;
  password[1] =b;
  
  
  for(w=48; w<=57; w++){
    //for(x=48; x<=57; x++){
     for(y=48; y<=57; y++){
      //for(z=48; z<=57; z++){
	password[2] = w;
        //password[2] = x;
        password[3] = y;
        //password[5] = z;

        if(is_a_match(password)) {
        printf("password found: %s\n", password);
      } else {
        //printf("tried: %s\n", password);		  
      }
}
}
}



int time_diff(struct timespec *start, 
                    struct timespec *finish, 
                    long long int *difference) {
  long long int ds =  finish->tv_sec - start->tv_sec; 
  long long int dn =  finish->tv_nsec - start->tv_nsec; 

  if(dn < 0 ) {
    ds--;
    dn += 1000000000; 
  } 
  *difference = ds * 1000000000 + dn;
  return !(*difference > 0);
}


int main(){

  struct timespec start, finish;   
  long long int time_elapsed;

  clock_gettime(CLOCK_MONOTONIC, &start);

  kernel <<<26, 26>>>();
  cudaThreadSynchronize();

  
  
  
  

  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_diff(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed, 
	(time_elapsed/1.0e9));

  return 0;
}


