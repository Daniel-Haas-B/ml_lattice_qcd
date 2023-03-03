/*
   Program to solve the two-dimensional Pott model
   with zero external field.
   The coupling constant J = 1
   Boltzmann's constant = 1, temperature has thus dimension energy
   Metropolis sampling is used. Periodic boundary conditions.
*/

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "lib.h"
using namespace std;

ofstream ofile;

// inline function for periodic boundary conditions
inline int periodic(int i, int limit, int add)
{
  return (i + limit + add) % (limit);
}
// Function to read in data from screen
void input(int &, int &, double &, double &, double &, int &);
// Function to initialise energy and magnetization
void initialize(int, double, int **, double &, double &, int, long &);
// The metropolis algorithm
void Metropolis(int, long &, int **, double &, double &,
                double, int, double *);
// prints to file the results of the calculations
void output(int, int, double, double *, int);

int main(int argc, char *argv[])
{
  char *outfilename;
  long idum;
  int **spin_matrix, n_spins, mcs, q;
  double w[9], average[5], initial_temp, final_temp, E, M,
      temp_step;

  // Read in output file, abort if there are too
  // few command-line arguments

  if (argc <= 1)
  {
    cout << "Bad Usage: " << argv[0] << " read also output file on same line" << endl;
    exit(1);
  }
  else
  {
    outfilename = argv[1];
  }
  ofile.open(outfilename);

  // Read in initial values such as size of lattice, temp and cycles
  input(n_spins, mcs, initial_temp, final_temp, temp_step, q);
  spin_matrix = (int **)matrix(n_spins, n_spins, sizeof(int));
  idum = -time(NULL); // random starting point
  for (double temperature = initial_temp; temperature <= final_temp;
       temperature += temp_step)
  {
    //    initialise energy and magnetization
    E = M = 0.;
    /* setup array for possible energy changes.
       Note the differende in the def for q=2.
    */
    if (q > 2)
    {
      for (int de = -4; de <= 4; de++)
        w[de + 4] = 0.;
      for (int de = -4; de <= 4; de++)
        w[de + 4] = exp(-de / temperature);
    }
    if (q == 2)
    {
      for (int de = -4; de <= 4; de++)
        w[de + 4] = 0.;
      for (int de = -4; de <= 4; de += 2)
        w[de + 4] = exp(-de / temperature);
    }
    // initialise array for expectation values
    for (int i = 0; i < 5; i++)
      average[i] = 0.;
    initialize(n_spins, temperature, spin_matrix, E, M, q, idum);
    // start Monte Carlo computation
    for (int cycles = 1; cycles <= mcs; cycles++)
    {
      Metropolis(n_spins, idum, spin_matrix, E, M, temperature, q, w);
      // update expectation values
      average[0] += E;
      average[1] += E * E;
      average[2] += M;
      average[3] += M * M;
      average[4] += fabs(M);
    }
    // print results
    output(n_spins, mcs, temperature, average, q);
    cout << temperature << endl;
  }
  free_matrix((void **)spin_matrix); // free memory
  ofile.close();                     // close output file
  return 0;
}

// read in input data
void input(int &n_spins, int &mcs, double &initial_temp,
           double &final_temp, double &temp_step, int &q)
{
  ifstream fin("pott_data.dat", ios::in);
  fin >> mcs;
  fin >> n_spins;
  fin >> initial_temp;
  fin >> final_temp;
  fin >> temp_step;
  fin >> q;

} // end of function read_input

// function to initialise energy, spin matrix and magnetization
void initialize(int n_spins, double temperature, int **spin_matrix,
                double &E, double &M, int q, long &idum)
{
  // double r = ran1(&idum);
  //  setup spin matrix and intial magnetization
  for (int y = 0; y < n_spins; y++)
  {
    for (int x = 0; x < n_spins; x++)
    {
      spin_matrix[y][x] = 1; // Spin orientation for ground state

      // Definition of the Magnetisation
      M += (double)spin_matrix[y][x];
    }
  }
  // setup initial energy
  for (int y = 0; y < n_spins; y++)
  {
    for (int x = 0; x < n_spins; x++)
    {
      if (spin_matrix[y][x] == spin_matrix[periodic(y, n_spins, -1)][x])
      {
        E = E - 1.;
      }
      if (spin_matrix[y][x] == spin_matrix[y][periodic(x, n_spins, -1)])
      {
        E = E - 1.;
      }
    }
  }
  // cout << E << endl;
} // end function initialise

void Metropolis(int n_spins, long &idum, int **spin_matrix, double &E,
                double &M, double temperature, int q, double *w)
{
  // loop over all spins
  int a, b;
  int deltaE;
  int e_f = 0, e_e = 0;
  double s = q * ran1(&idum);
  int ix = (int)(ran1(&idum) * n_spins);
  int iy = (int)(ran1(&idum) * n_spins);

  b = spin_matrix[iy][ix];
  a = spin_matrix[iy][ix];

  do
  {
    a = (int)(ran1(&idum) * q) + 1;
  } while (a == b);

  // Calculating the energy difference between spin a and b.
  if (b == spin_matrix[iy][periodic(ix, n_spins, -1)])
    e_f--;
  if (b == spin_matrix[periodic(iy, n_spins, -1)][ix])
    e_f--;
  if (b == spin_matrix[iy][periodic(ix, n_spins, 1)])
    e_f--;
  if (b == spin_matrix[periodic(iy, n_spins, 1)][ix])
    e_f--;

  if (a == spin_matrix[iy][periodic(ix, n_spins, -1)])
    e_e--;
  if (a == spin_matrix[periodic(iy, n_spins, -1)][ix])
    e_e--;
  if (a == spin_matrix[iy][periodic(ix, n_spins, 1)])
    e_e--;
  if (a == spin_matrix[periodic(iy, n_spins, 1)][ix])
    e_e--;

  deltaE = (e_e - e_f);

  if (deltaE <= 0)
  {
    spin_matrix[iy][ix] = a; // flip one spin and accept
    M += (double)(a - b);    // new spin config
    E += (double)deltaE;
  }
  else if (ran1(&idum) <= w[deltaE + 4])
  {
    spin_matrix[iy][ix] = a; // flip one spin and accept
    M += (double)(a - b);    // new spin config
    E += (double)deltaE;
  }

} // end of Metropolis sampling over spins

void output(int n_spins, int mcs, double temperature,
            double *average, int q)
{

  double norm = 1 / ((double)(mcs)); // divided by total number of cycles

  double Eaverage = average[0] * norm;
  double E2average = average[1] * norm;
  double Maverage = average[2] * norm;
  double M2average = average[3] * norm;
  double Mabsaverage = average[4] * norm;
  // all expectation values are per spin,
  // divide by 1/n_spins/n_spins
  double Evariance = (E2average - Eaverage * Eaverage) / n_spins / n_spins;
  double Mvariance = (M2average - Mabsaverage * Mabsaverage) / n_spins / n_spins;
  ofile << setiosflags(ios::showpoint | ios::uppercase);
  ofile << setw(15) << setprecision(8) << temperature;
  ofile << setw(15) << setprecision(8) << Eaverage / n_spins / n_spins;
  ofile << setw(15) << setprecision(8) << Evariance / temperature / temperature;
  ofile << setw(15) << setprecision(8) << Maverage / n_spins / n_spins;
  ofile << setw(15) << setprecision(8) << Mvariance / temperature;
  ofile << setw(15) << setprecision(8) << Mabsaverage / n_spins / n_spins << endl;
} // end output function
