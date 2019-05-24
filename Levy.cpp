#include <QCoreApplication>
#include<math.h>
#include <iostream>
#include <QDebug>
#include <gsl/gsl_randist.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <fstream>
#include <math.h>
using namespace  std;
int rand_position;
double rand1;

double randf(double f)
{
    return pow((double)(rand())/RAND_MAX,(float)-(1)/1.5);
       //return pow((double)rand1,(float)-(1)/1.5);
}
int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    const gsl_rng_type * T;
    const gsl_rng_type * T2;
     gsl_rng * rx;
     gsl_rng * ry;

     //FILE *f=fopen ("output.txt","w");
     /*for n=2:s;
         theta=rand*2*pi;
         f=rand^(-1/alpha);
         x(n)=x(n-1)+f*cos(theta);
         y(n)=y(n-1)+f*sin(theta);
     end;*/

     int i, n = 1000;
    double*x = new double[n];
    double*y = new double[n];
    double theta,f;

     gsl_rng_env_setup();

     T = gsl_rng_default;
     T2 = gsl_rng_mt19937;
     rx = gsl_rng_alloc (T);
     ry = gsl_rng_alloc (T2);
    x[0] = 0;
    y[0] = 0;
         for (i = 1; i < n; i++)
         {
             x[i] = 0;
             y[i] = 0;
         }
     for (i = 1; i < n; i++)
       {
         rand1 = (double)(rand())/RAND_MAX ;
         theta=rand1*2*3.1415;
         f=randf(f);
         if (f >= 3.f || f<=1.f)
         {
             i--;
         }

         x[i]=(x[i-1]+f*cos(theta));
         y[i]=(y[i-1]+f*sin(theta));
         printf ("%f %f\n", x[i],y[i]);
       }

          for (i = 1; i < n; i++)
          {
             x[i] = x[i] * (-2.f);
              y[i] = y[i] * (-2.f);
          }

     gsl_rng_free (rx);
      gsl_rng_free (ry);

      fstream fx;
      fstream fy;
      fx.open("/home/hunk/Levy/x", fstream::in | fstream::out);
      fy.open("/home/hunk/Levy/y", fstream::in | fstream::out);
      if(fx == NULL || fy ==NULL)
      {
          cout << "error!";
          return -1;
      }

      for(int i=0; i<n; i++)    // ??? ???
      {
          fx << x[i]<< endl;
          fy <<y[i] << endl;
      }

      fx.close();
      fy.close();
    return a.exec();
}


