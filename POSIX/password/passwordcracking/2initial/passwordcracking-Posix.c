#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <time.h>
#include <pthread.h>

/***********************************************************************
*******
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer. Your personalised data set is included in the code. 

  Compile with:
    cc -o passwordcracking-Posix passwordcracking-Posix.c -lcrypt -pthread

  If you want to analyse the results then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:

    ./passwordcracking-Posix > results.txt

*** Nirdeshika KC ***

  Dr Kevan Buckley, University of Wolverhampton, 2018
************************************************************************
******/
int count = 0; //count the solution explored

int n_passwords = 4;

char *encrypted_passwords[] = {

	"$6$KB$6SsUGf4Cq7/Oooym9WWQN3VKeo2lynKV9gXVyEG4HvYy1UFRx.XAye89TLp/OTcW7cGpf9UlU0F.cK/S9CfZn1",

	"$6$KB$.bVspZYQofaBc4KhsjlqZSxu4R7r7mH7.Q/uCYlJ.3nRV2x5Jz.TKYX6Aa97sUZhTjmN3rett7GrCFr3aO3uR/",

	"$6$KB$Z/wMw97.MbsjxFvFZacJuQYiyegBaODZm0KBZFiXMVhWxywV/LC5UD9YRr0WYUnzLcc65r886iJwORXyEf2eY.",

	"$6$KB$XxG6dvtXywYrGaKzTwfm0TvaSlbpVQjFUCbW2KofszOd7vmECs7Eqi4Z03rCh8kY6K9t3NR1p6o8AowNBlu8R1"
      };

/**
 Required by lack of standard function in C.   
*/

void substr(char *dest, char *src, int start, int length){
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the 
 start of the line. Note that one of the most time consuming operations that 
 it performs is the output of intermediate results, so performance experiments 
 for this kind of program should not include this. i.e. comment out the
 printfs.
*/

void *pwdcrack1(void *salt_and_encrypted){
  int a, b, c;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space for \0
  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password

  substr(salt, salt_and_encrypted, 0, 6);
  for(a='A'; a<='M'; a++){
    for(b='A'; b<='Z'; b++){
      for(c=0; c<=99; c++){
        sprintf(plain, "%c%c%02d", a,b,c); 
        enc = (char *) crypt(plain, salt);
        count++;
        if(strcmp(salt_and_encrypted, enc) == 0){
          printf("#%-8d%s %s\n", count, plain, enc);
        } //else {
          //printf(" %-8d%s %s\n", count, plain, enc);
        //}
      }
    }

  }
  pthread_exit(NULL);
}

void *pwdcrack2(void *salt_and_encrypted){
  int a, b, c;     // Loop counters
  char salt[7];    // String used in hashing the password. Need space for \0
  char plain[7];   // The combination of letters currently being checked
  char *enc;       // Pointer to the encrypted password

  substr(salt, salt_and_encrypted, 0, 6);
	
  for(a='N'; a<='Z'; a++){
    for(b='A'; b<='Z'; b++){
      for(c=0; c<=99; c++){
        sprintf(plain, "%c%c%02d", a,b,c); 
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

pthread_exit(NULL);
}

//function to calculate running time of crack function
int time_diff(struct timespec *start, struct timespec *finish,
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


int main(int argc, char *argv[]){
  int i;
  struct timespec start, finish;   
  long long int time_elapsed;

  pthread_t thread_1;
  pthread_t thread_2;
  int t1, t2;

  clock_gettime(CLOCK_MONOTONIC, &start);
  for(int i =0; i<n_passwords; i++){
  t1 = pthread_create(&thread_1, NULL, pwdcrack1,(void *)encrypted_passwords[i]);
  if(t1){
  printf("Thread creation failed: %d\n", t1);
  }
 }
  for(int i =0; i<n_passwords; i++){
  t2 = pthread_create(&thread_2, NULL, pwdcrack2,(void *)encrypted_passwords[i]);
  if(t2){
  printf("Thread creation failed: %d\n", t2);
  }
 }
  

  pthread_join(thread_1, NULL);
  pthread_join(thread_2, NULL);

  clock_gettime(CLOCK_MONOTONIC, &finish);
  printf("%d solutions explored\n", count);
  time_diff(&start, &finish, &time_elapsed);
  printf("Time elapsed was %lldns\n", time_elapsed);

 pthread_exit(NULL);
}


