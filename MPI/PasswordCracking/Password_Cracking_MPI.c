#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>
#include <time.h>
/***********************************************************************
  Compile with:
    mpicc -o Password_Cracking_MPI Password_Cracking_MPI.c -lcrypt

  we can redirect result by applying

  To run:
    mpirun -n 3 ./Password_Cracking_MPI > results.txt

*** Nirdeshika KC ***

************************************************************************
******/
int n_passwords = 4;

char *encrypted_passwords[] = {

"$6$KB$K7t0mdm2c1m4doUgHUSblA18lwf/kPFlj7yOsKARgRw.0L.NEPHns/sjulzoaklEh.pPicgRWxUHvVzDFVaHt0",
"$6$KB$RcP4jqk.NPAuCZkco5GV/twSgT.oh3uh5d2uJOV34KbI6Wb7998mDlHb2t9kK0p0lx/JcnsSldiQbmnolcNNu1",
"$6$KB$T3.N1e1n.SeEaLXbN6rWf77RuYjaKFH2mE2z3tbqKn8SZnUx0/DbkI/YatECIuFmiLzqJ0nT.lFay8D0/6qd00",
"$6$KB$joc9OPGjxifQU65CL8CBjrzktkRaTaCh6V4EGA3GDVQNU1o6SXjZkPEzFv7sY6AKVKygbtHTPQ8sW.gvlkd0x."
};

void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All
combinations
 that are tried are displayed and when the password is found, #, is put
at the 
*/

void func1(char *salt_and_encrypted){
  int a, y, c,d;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space

  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password
  int count = 0;   // The number of combinations explored so far

  substr(salt, salt_and_encrypted, 0, 6);

  for(a='A'; a<='M'; a++){
    for(y='A'; y<='Z'; y++){
      for(c=0; c<=99; c++){
         for(d=0; d<=99;d++){
      	//printf("Instance 1");
        sprintf(plain, "%c%c%02d%02d", a, y, c,d); 
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(salt_and_encrypted, enc) == 0){
          printf("#%-8d%s %s\n", count, plain, enc);
        } else {
          //printf(" %-8d%s %s\n", count, plain, enc);

         }
        }
      }
    }
  }
  printf("%d solutions explored\n", count);
}

void func2(char *salt_and_encrypted){
  int a, y, c,d;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space

  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password
  int count = 0;   // The number of combinations explored so far

  substr(salt, salt_and_encrypted, 0, 6);

  for(a='N'; a<='Z'; a++){
    for(y='A'; y<='Z'; y++){
      for(c=0; c<=99; c++){
	for(d=0; d<=99;d++){
         //printf("Instance 2");
        sprintf(plain, "%c%c%02d%02d", a, y, c,d); 
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(salt_and_encrypted, enc) == 0){
          printf("#%-8d%s %s\n", count, plain, enc);
        } else {
          //printf(" %-8d%s %s\n", count, plain, enc);
	}
        }
      }
    }
  }
  printf("%d solutions explored\n", count);
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
 

int size, rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(size != 3) {
    if(rank == 0) {
      printf("This program needs to run on exactly 3 processes\n");
    }
  } else {
    if(rank ==0){
      int a;
      int y;
      int i;
	MPI_Send(&a, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);  
        MPI_Send(&y, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
	
    } else {
      if(rank == 1){
	int i;
        int number = rank + 10;
      	MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	for ( i = 0; i<n_passwords;i<i++){
		func1(encrypted_passwords[i]);
	}
      }
	else if(rank == 2){
	int i;
      	int number = rank + 10;
      	MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	for ( i = 0; i<n_passwords;i<i++){
		func2(encrypted_passwords[i]);
	}
	}
    }
}  
  MPI_Finalize(); 

  clock_gettime(CLOCK_MONOTONIC, &finish);
  time_diff(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns or %0.9lfs\n", time_elapsed, 
	(time_elapsed/1.0e9));


  return 0;
}


