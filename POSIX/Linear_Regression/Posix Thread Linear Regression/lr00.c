#include <stdio.h>

/******************************************************************************
 * This is the first program in a series on linear regression. The purpose of
 * this program is to enable the understanding of how the data to be processed 
 * is stored and how to access it. The program simply writes the comma delimited
 * data points to stdout. In order to understand what the data looks like it 
 * should be redirected into a text file and plotted using a scatter graph in 
 * a spreadsheet. You should analyse the graph and estimate the slope (m) and 
 * intercept (c) of the optimum regression line.
 * 
 * To compile:
 *   cc -o lr00 lr00.c
 * 
 * To run and direct the output into a file:
 *   ./lr00 > scatter.csv
 * 
 * To load results into a spreadsheet:
 *   libreoffice lr00_results.csv
 * 
 * Dr Kevan Buckley, University of Wolverhampton, 2018
 *****************************************************************************/

typedef struct point_t {
  double x;
  double y;
} point_t;

int n_data = 1000;

point_t data[];

void print_data() {
  int i;
  for(i=0; i<n_data; i++) {
    printf("%0.2lf,%0.2lf\n", data[i].x, data[i].y);
  }
}

int main() {
  print_data();
  return 0;
}

point_t data[] = {
  {65.43,96.04},{78.25,108.40},{78.72,111.38},{45.68,65.03},
  {77.35,123.59},{65.53,82.25},{82.01,119.20},{73.66,107.79},
  {69.44,109.36},{66.65,101.25},{65.78,91.96},{82.75,95.96},
  {82.43,124.20},{79.83,109.90},{78.50,114.35},{26.76,55.22},
  {14.94,52.29},{77.29,96.55},{96.13,128.05},{10.57,36.50},
  {73.48,93.15},{24.37,55.63},{78.62,119.22},{58.55,87.03},
  {50.67,83.61},{21.41,38.96},{70.53,97.46},{38.86,66.17},
  {85.48,110.52},{64.65,96.78},{19.98,65.06},{96.36,129.83},
  {77.54,136.78},{32.36,56.96},{ 2.73,19.36},{14.08,40.70},
  {30.49,56.81},{68.98,97.09},{18.74,58.74},{55.40,73.82},
  {35.15,61.00},{58.85,99.14},{33.63,74.62},{37.47,62.17},
  {59.16,92.51},{36.94,56.06},{34.78,72.97},{88.72,126.02},
  {24.77,40.83},{12.04,45.12},{68.67,108.00},{88.13,118.81},
  {53.35,91.49},{81.14,97.23},{47.85,74.17},{25.65,60.28},
  {98.44,139.54},{42.69,73.25},{33.25,66.00},{ 0.25,16.13},
  {59.64,100.67},{ 1.08,18.06},{27.65,59.71},{51.41,70.92},
  {37.06,58.25},{46.75,74.02},{44.11,70.02},{17.25,35.57},
  {61.03,103.51},{ 2.49,15.85},{ 1.82,26.54},{84.70,117.49},
  {75.52,124.77},{42.29,68.75},{ 3.80,31.36},{11.72,32.71},
  { 4.23,30.00},{91.03,126.98},{27.71,59.91},{25.29,52.08},
  { 1.37,14.37},{43.66,70.35},{79.88,114.95},{92.60,127.80},
  {63.15,88.63},{93.37,133.49},{19.01,42.26},{99.88,146.22},
  {87.62,117.88},{81.72,119.57},{34.85,62.48},{64.76,76.47},
  {97.90,122.16},{65.72,92.58},{59.37,88.09},{18.29,65.76},
  {17.16,39.57},{56.28,86.30},{40.31,72.30},{29.63,64.11},
  {87.89,116.45},{84.60,123.67},{42.63,64.44},{73.34,106.20},
  {52.80,99.28},{49.63,77.93},{96.81,133.36},{10.85,28.60},
  {98.63,131.71},{75.31,106.16},{82.53,113.20},{86.36,119.43},
  {78.91,123.68},{13.52,52.73},{82.75,114.46},{88.24,105.06},
  {36.02,74.28},{23.99,57.80},{23.69,38.33},{51.21,73.16},
  {40.39,56.89},{ 9.70,38.25},{ 1.56,31.13},{66.57,107.37},
  {99.73,130.76},{52.76,65.48},{94.48,129.26},{80.19,97.31},
  {26.92,35.32},{88.23,119.83},{86.78,104.74},{36.24,51.68},
  {27.59,64.27},{58.30,74.17},{72.64,113.30},{13.62,45.02},
  {30.36,46.09},{71.66,105.65},{73.46,110.49},{ 8.36,23.79},
  {23.13,45.81},{39.10,67.65},{36.08,54.15},{16.86,39.04},
  {18.15,44.28},{ 7.54,26.67},{27.05,54.09},{14.85,48.06},
  {37.63,58.24},{15.64,50.08},{93.18,136.70},{76.72,98.06},
  {49.26,73.97},{18.04,32.34},{ 3.00,29.26},{40.14,68.94},
  {91.52,100.02},{29.79,55.05},{72.88,100.14},{24.18,57.84},
  {88.45,122.85},{14.32,33.64},{ 7.90,30.82},{39.50,71.32},
  {85.01,125.64},{15.28,56.70},{43.83,65.75},{77.51,103.25},
  {78.63,127.01},{48.57,73.43},{54.64,99.83},{82.96,105.72},
  {98.17,131.44},{38.36,71.37},{44.65,90.56},{54.67,69.44},
  {83.22,110.90},{ 2.98,30.01},{47.65,79.96},{21.26,60.09},
  { 1.37,10.02},{10.52,32.32},{28.70,52.77},{84.58,114.35},
  {35.49,75.53},{98.87,127.25},{23.30,34.55},{95.77,142.36},
  { 5.73,15.41},{55.32,70.43},{84.95,116.98},{59.30,69.34},
  {91.33,112.38},{86.29,115.55},{98.87,128.66},{50.32,66.02},
  {63.24,85.58},{84.85,107.68},{85.35,117.98},{13.02,48.96},
  {96.11,124.60},{26.84,52.19},{91.36,134.26},{11.53,39.49},
  { 4.66,46.94},{66.28,111.64},{88.20,137.84},{47.49,74.54},
  {78.41,106.24},{21.60,37.84},{15.79,42.63},{61.08,98.27},
  {13.37,41.26},{16.71,47.18},{ 8.45,56.30},{11.69,38.21},
  {96.97,139.19},{ 2.07,27.80},{61.46,90.87},{68.66,105.36},
  {44.28,74.80},{ 0.55,29.49},{34.18,54.35},{58.45,117.24},
  {88.42,116.32},{72.33,126.82},{50.91,71.20},{98.54,119.96},
  {87.83,117.14},{63.92,113.55},{17.09,63.72},{42.12,78.29},
  {46.47,65.33},{33.84,35.84},{43.83,84.86},{89.21,109.88},
  {73.35,107.44},{82.55,87.14},{21.76,23.39},{47.36,93.55},
  {83.81,120.86},{27.30,52.31},{86.36,115.11},{17.73,49.48},
  {98.31,133.55},{ 3.55,43.23},{52.47,81.89},{ 9.99,35.61},
  {60.85,78.55},{93.87,152.56},{16.35,11.19},{76.60,115.01},
  {10.04,40.79},{98.56,124.98},{61.76,88.70},{59.58,94.99},
  {51.53,81.54},{49.48,86.59},{68.05,90.24},{46.37,69.16},
  {76.71,102.36},{18.67,37.94},{ 8.41,41.65},{18.38,54.01},
  {65.48,106.19},{61.57,80.77},{75.23,100.17},{59.61,92.43},
  {25.49,45.73},{81.03,94.90},{43.47,52.58},{ 4.26,29.52},
  {61.00,99.11},{97.72,125.98},{87.26,114.19},{75.97,103.71},
  {53.12,88.11},{ 0.94,33.56},{10.81,41.88},{50.95,75.96},
  { 7.08,28.44},{87.77,122.44},{ 7.86,20.87},{66.09,99.55},
  {78.06,111.26},{19.11,46.12},{ 9.64,38.76},{84.08,113.09},
  {55.81,104.50},{26.64,52.97},{85.53,130.35},{90.80,119.84},
  { 5.21,18.83},{72.15,120.70},{10.34,40.48},{70.81,105.72},
  {62.22,75.33},{ 1.37,14.13},{79.69,117.90},{98.12,125.74},
  {38.72,65.47},{76.00,114.15},{67.28,94.16},{42.06,66.65},
  {19.79,43.25},{20.27,46.00},{17.96,66.11},{45.17,77.51},
  {89.83,139.80},{95.07,125.52},{38.32,71.11},{17.51,43.73},
  {40.47,63.47},{55.73,58.82},{82.61,117.14},{68.05,96.95},
  {31.17,65.51},{ 3.17,37.52},{10.88,36.56},{68.91,94.69},
  {98.34,122.76},{98.41,112.94},{79.24,116.44},{72.52,118.27},
  {64.70,80.36},{19.97,57.52},{52.43,61.62},{81.16,121.53},
  {81.61,99.42},{59.33,66.61},{20.95,40.47},{72.90,110.05},
  {71.17,94.64},{13.80,43.75},{64.13,101.71},{44.72,64.29},
  {24.56,51.65},{12.35,37.58},{59.84,72.72},{16.70,25.09},
  {82.80,117.96},{55.08,93.33},{54.03,93.50},{93.95,116.23},
  {24.20,39.73},{19.69,46.12},{60.45,83.29},{51.48,71.84},
  {86.97,110.41},{72.54,113.04},{ 9.99,36.68},{90.18,122.44},
  {52.78,87.33},{16.62,50.11},{48.89,78.21},{89.12,130.87},
  {97.08,113.75},{26.22,72.95},{23.07,66.48},{84.91,107.37},
  {60.74,94.24},{ 7.03,31.43},{58.91,91.62},{94.08,103.14},
  {63.21,78.44},{30.23,44.26},{12.00,28.46},{44.70,71.09},
  {12.26,36.71},{32.79,67.98},{67.87,121.36},{15.84,36.57},
  {44.07,65.38},{37.69,55.65},{41.13,66.42},{36.61,43.98},
  {24.33,61.73},{55.30,85.52},{33.69,76.36},{12.08,35.71},
  { 5.17,28.11},{39.95,64.72},{90.86,118.21},{26.99,66.84},
  {47.38,80.16},{49.96,83.14},{ 8.99,21.01},{60.11,94.14},
  {29.93,47.80},{29.75,64.31},{89.40,95.36},{12.40,24.21},
  { 5.48,21.33},{25.40,63.64},{57.50,81.72},{34.89,75.33},
  {31.18,36.22},{ 4.33,18.06},{71.76,108.30},{41.85,63.87},
  {95.30,141.93},{23.44,64.21},{67.54,95.80},{79.98,119.06},
  {76.13,95.39},{12.66,29.00},{90.55,106.60},{63.40,106.73},
  {46.14,73.35},{20.42,45.03},{67.45,105.49},{79.52,110.42},
  {46.62,89.75},{ 1.77,32.43},{39.78,71.17},{60.25,90.03},
  {62.38,101.37},{27.19,56.11},{25.78,43.84},{48.31,70.38},
  { 6.78,28.42},{73.63,126.28},{65.27,93.47},{11.75,30.36},
  {40.81,53.58},{31.04,70.61},{60.29,106.56},{43.56,52.78},
  {85.34,116.62},{39.93,66.51},{77.51,119.55},{48.30,84.91},
  {48.15,101.21},{70.10,104.38},{63.64,85.06},{65.69,89.18},
  {80.52,90.74},{21.06,56.55},{27.09,57.25},{86.90,119.51},
  {78.59,129.97},{90.71,131.38},{64.34,98.98},{41.05,93.36},
  {45.65,69.45},{36.51,72.02},{93.13,124.24},{89.94,136.28},
  {43.77,78.16},{19.42,39.88},{54.15,75.46},{ 8.98,18.07},
  {63.18,70.93},{59.01,97.11},{54.85,68.04},{46.21,75.25},
  {48.74,76.05},{20.32,40.65},{18.36,20.74},{25.58,51.78},
  {48.78,64.78},{20.57,41.29},{70.88,96.75},{17.90,37.35},
  {56.64,79.01},{44.68,83.08},{27.30,61.30},{70.35,87.64},
  {46.25,61.42},{74.95,114.30},{24.03,52.25},{89.02,118.71},
  {71.26,114.02},{ 3.19, 6.55},{71.39,108.12},{81.59,106.10},
  {80.13,103.97},{90.74,134.59},{70.44,99.08},{43.52,70.52},
  {79.43,102.59},{ 4.07,11.72},{22.97,39.77},{70.73,113.21},
  {29.65,57.14},{33.44,70.49},{79.19,110.72},{89.44,129.08},
  {18.18,48.93},{57.45,90.32},{72.90,99.33},{84.50,103.84},
  {23.26,66.97},{17.93,39.49},{73.05,84.20},{ 3.25,38.76},
  {56.67,81.50},{68.39,87.17},{16.77,49.24},{58.73,99.01},
  {31.16,54.19},{71.44,107.43},{92.11,124.32},{38.41,72.42},
  {43.10,67.11},{63.12,95.81},{65.38,90.26},{60.74,88.50},
  {94.88,132.75},{45.29,66.05},{35.48,50.51},{48.95,84.91},
  {73.74,122.01},{32.60,66.29},{33.77,60.09},{77.68,118.25},
  {79.24,129.79},{63.08,104.85},{73.84,112.07},{89.23,122.07},
  { 5.83, 8.04},{68.36,92.83},{90.28,115.20},{48.97,67.54},
  {87.77,109.38},{63.51,98.33},{76.02,106.09},{30.69,63.72},
  {65.33,112.84},{80.59,96.05},{99.45,134.07},{ 4.50,20.16},
  {74.85,101.22},{34.58,47.82},{75.71,95.54},{45.09,63.56},
  {78.56,96.99},{19.58,45.26},{37.48,50.63},{16.67,28.13},
  {60.39,100.48},{43.19,73.17},{63.69,97.47},{69.93,108.64},
  { 3.14,23.10},{42.73,78.29},{59.69,90.33},{ 3.41,34.87},
  {47.47,104.81},{71.89,100.47},{98.72,134.53},{62.69,97.89},
  {76.09,108.57},{25.11,35.58},{45.75,88.45},{52.23,84.43},
  {77.47,102.92},{35.31,66.38},{43.01,84.64},{11.96,37.89},
  {21.95,59.77},{71.82,101.48},{98.81,137.88},{23.12,41.14},
  {100.00,133.69},{44.78,75.86},{75.38,111.29},{18.62,46.95},
  { 0.65,28.61},{17.48,48.60},{43.44,76.02},{61.48,88.40},
  {78.48,107.07},{24.08,48.39},{ 8.16,43.49},{15.55,35.87},
  {21.19,48.10},{18.37,32.91},{44.99,67.86},{33.15,49.92},
  {45.11,60.65},{96.49,122.70},{63.35,88.83},{82.71,104.50},
  {17.54,32.70},{89.71,134.71},{85.30,123.30},{ 4.33,32.52},
  {86.67,130.26},{63.07,70.32},{94.60,132.36},{79.15,120.56},
  {57.00,80.60},{11.08,41.01},{74.06,108.78},{32.09,64.30},
  {93.54,132.41},{87.09,143.95},{53.79,88.06},{85.42,109.65},
  {20.82,35.12},{53.96,85.52},{98.82,142.35},{86.80,135.28},
  {66.08,115.33},{46.51,80.94},{22.60,46.60},{13.94,37.29},
  {54.41,92.01},{ 7.39,34.98},{42.37,58.68},{25.40,48.68},
  {75.92,93.04},{85.27,108.39},{88.88,126.52},{52.81,89.85},
  {56.09,64.72},{52.65,85.57},{59.43,98.18},{28.77,57.26},
  {93.93,130.41},{86.74,113.84},{79.32,106.88},{97.30,128.12},
  {64.81,86.18},{83.14,123.54},{18.23,58.94},{31.45,76.07},
  {53.28,77.89},{ 2.51,17.07},{37.03,70.75},{57.14,95.33},
  {32.96,53.65},{95.55,125.42},{93.74,136.19},{83.49,113.23},
  {20.94,40.63},{68.39,103.50},{ 7.77,22.20},{64.14,86.81},
  {77.11,100.89},{18.72,38.83},{94.24,128.89},{ 7.20,32.90},
  { 5.29,46.32},{76.46,105.72},{40.12,61.02},{68.86,113.33},
  {12.55,29.69},{51.68,75.54},{54.18,74.44},{49.09,85.28},
  {70.68,116.11},{ 5.21,32.12},{74.33,97.88},{19.78,39.24},
  {35.34,53.13},{20.63,41.53},{66.29,90.73},{31.58,63.58},
  {50.91,90.00},{73.63,108.26},{68.68,112.53},{68.00,111.88},
  {89.89,119.53},{11.59,52.83},{79.81,121.08},{43.68,68.10},
  {90.38,132.30},{95.11,121.37},{12.29,43.36},{62.71,111.83},
  {41.09,78.95},{86.65,128.11},{66.40,97.35},{ 8.74,31.34},
  {65.15,98.79},{72.44,104.23},{80.62,112.48},{40.87,75.50},
  {43.31,72.91},{60.78,78.81},{79.18,117.95},{22.56,64.82},
  { 9.90,29.63},{88.77,124.78},{44.22,78.91},{55.94,85.18},
  {60.03,92.09},{47.79,86.12},{84.50,112.16},{34.06,62.36},
  {84.18,124.81},{20.61,39.74},{14.40,45.80},{80.19,102.94},
  {13.45,45.55},{95.39,131.97},{92.31,127.08},{73.67,94.93},
  {91.15,117.45},{24.81,42.42},{99.76,135.94},{34.31,76.61},
  {10.30,44.89},{76.66,101.73},{92.64,124.60},{72.77,101.52},
  {29.72,52.43},{31.43,65.50},{72.97,104.08},{95.82,129.36},
  {13.94,36.88},{94.65,118.47},{91.09,117.33},{31.93,64.49},
  {54.58,95.28},{94.74,128.42},{24.21,44.89},{97.86,125.22},
  {43.77,71.13},{48.26,96.59},{64.01,109.44},{98.97,127.89},
  {57.11,78.95},{ 5.81,49.23},{38.25,58.44},{ 3.28,38.30},
  {31.82,69.33},{24.83,58.38},{53.43,74.72},{75.14,100.07},
  {14.32,36.05},{91.69,130.36},{ 2.85,29.31},{64.16,79.01},
  {76.58,93.49},{31.13,62.22},{40.60,77.08},{98.07,135.07},
  {90.30,131.58},{31.85,61.20},{68.86,107.89},{20.89,40.46},
  {11.31,40.33},{87.59,123.89},{40.21,63.19},{ 7.16,18.01},
  {26.61,57.16},{89.38,120.44},{25.52,62.50},{48.59,84.42},
  {82.92,109.09},{11.02,36.13},{36.07,74.83},{47.62,80.03},
  {97.85,116.09},{76.07,101.41},{60.88,80.08},{55.88,87.91},
  {43.25,65.49},{49.52,82.07},{46.91,69.46},{80.93,121.24},
  {26.19,59.22},{35.07,81.81},{82.06,112.43},{ 5.80,15.34},
  {27.66,48.92},{78.26,118.06},{73.89,104.57},{44.14,71.01},
  {65.88,95.44},{72.69,96.40},{16.05,51.08},{10.52,42.74},
  {37.10,42.03},{68.96,94.22},{82.88,139.00},{36.28,59.46},
  {68.80,87.80},{ 5.18,47.34},{81.13,108.50},{85.30,110.25},
  {61.96,90.68},{57.72,103.73},{58.10,87.65},{42.01,76.07},
  {60.31,95.80},{85.61,117.77},{77.95,95.57},{59.61,98.20},
  { 0.81,39.56},{66.03,90.52},{15.84,33.57},{ 3.94,48.29},
  {30.81,53.20},{28.14,56.21},{70.42,119.75},{46.73,80.62},
  {63.27,88.36},{67.31,116.42},{92.41,127.50},{59.41,83.83},
  {14.50,47.52},{70.43,93.66},{ 5.53,19.87},{ 3.30,23.00},
  {91.83,134.55},{33.91,73.27},{41.14,51.63},{58.02,66.72},
  {16.97,41.13},{96.79,125.21},{52.16,76.21},{46.36,86.46},
  {16.53,40.92},{93.89,134.58},{64.32,85.44},{40.06,67.00},
  {11.94,18.37},{21.95,35.86},{59.09,107.14},{43.61,62.53},
  {79.79,112.43},{55.57,86.45},{64.99,92.10},{17.57,42.12},
  {14.42,44.71},{76.79,100.83},{57.71,94.19},{73.86,118.61},
  {70.01,86.22},{89.16,118.94},{85.27,120.52},{ 4.65,31.38},
  {47.46,79.26},{ 9.42,46.31},{21.82,49.14},{57.29,74.60},
  {23.56,54.57},{57.76,88.73},{84.47,128.13},{23.57,44.73},
  {46.63,81.23},{70.41,97.41},{97.25,117.89},{73.48,109.26},
  {77.30,127.55},{97.32,117.97},{24.64,58.46},{63.52,94.65},
  { 7.57,32.18},{98.34,144.21},{45.49,69.42},{67.19,91.72},
  {27.17,59.79},{94.55,125.80},{66.05,86.97},{23.28,66.11},
  {45.31,67.47},{13.06,31.36},{43.93,51.54},{45.10,80.82},
  {26.43,47.51},{48.61,73.74},{53.65,93.84},{35.27,51.51},
  {14.78,51.65},{42.25,46.06},{24.89,38.07},{29.53,63.39},
  { 0.72,27.10},{ 1.22,33.77},{89.35,115.27},{82.57,123.87},
  {31.93,53.20},{71.86,99.44},{16.08,47.99},{ 6.81,40.15},
  {78.37,109.21},{66.27,101.40},{33.92,52.94},{43.47,74.49},
  {20.77,72.57},{88.87,122.31},{31.24,57.07},{92.43,128.07},
  {14.84,43.93},{10.42,43.20},{84.78,105.45},{44.78,62.37},
  {66.01,102.99},{16.54,28.24},{ 9.06,58.88},{35.64,60.86},
  {20.05,29.33},{96.94,147.66},{73.26,113.66},{48.65,82.89},
  {44.30,64.28},{47.27,76.57},{57.73,93.05},{25.95,54.57},
  {29.45,48.82},{91.83,123.07},{69.12,100.21},{27.74,38.61},
  {37.77,68.89},{25.11,44.19},{74.20,122.71},{ 2.58,25.07},
  {71.83,113.30},{28.78,50.43},{95.89,121.16},{65.46,93.73},
  {41.20,72.53},{60.05,75.27},{ 8.79,33.20},{13.04,34.32},
  {20.79,51.77},{87.74,130.98},{41.45,73.72},{76.55,126.54},
  { 8.49,31.57},{50.52,92.82},{17.82,50.73},{47.85,75.15},
  {18.20,44.88},{64.15,85.50},{93.66,131.32},{72.52,101.03},
  {11.21,57.05},{50.43,81.28},{66.21,92.99},{29.37,61.11},
  {40.44,69.29},{15.44,50.49},{88.07,125.46},{10.84,38.31},
  {89.08,109.13},{13.62,40.52},{62.37,103.06},{62.72,96.01},
  {58.66,86.11},{65.82,84.74},{27.67,31.19},{48.35,100.21},
  { 7.73,33.38},{ 4.93,41.82},{41.85,69.57},{72.26,97.83},
  {82.50,105.47},{96.78,131.57},{98.16,122.86},{10.31,50.43},
  { 9.08,38.21},{79.61,122.11},{83.27,124.83},{74.88,105.97},
  {64.20,95.65},{ 6.32,24.17},{35.83,60.42},{19.38,40.02},
  {77.43,104.51},{20.36,33.85},{19.89,38.34},{47.18,76.05},
  {94.95,136.93},{64.39,93.42},{18.18,50.37},{85.08,123.16},
  {75.68,104.97},{ 1.27,21.07},{92.68,106.98},{91.32,133.52},
  {10.75,24.48},{35.52,66.23},{79.64,105.64},{35.15,56.99},
  {75.85,119.27},{91.69,117.28},{45.32,63.97},{24.66,35.01},
  {88.04,124.97},{96.43,118.28},{63.78,79.44},{70.96,111.55},
  {39.52,66.06},{42.15,73.93},{56.75,79.30},{86.09,112.30},
  {74.38,97.84},{50.36,80.08},{89.10,118.46},{17.61,46.82},
  {72.11,103.81},{ 5.14,24.15},{69.64,111.80},{61.30,98.92}
};
