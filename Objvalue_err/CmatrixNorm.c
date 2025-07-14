#include <stdio.h>
#include<math.h>
double Norminf(double x[],int n,int m);
double Norm1(double x[],int n,int m);
double NormF(double x[],int n,int m);
int main()
{
    double a,b,c,d,e,f;
    double x[6]={1,3,-2,-2,0,5},y[6]={4,3,1,-2,0,8};
    a=Norminf(x,3,2);    b=Norm1(x,3,2);    c=NormF(x,3,2);
    d=Norminf(y,2,3);    e=Norm1(y,2,3);    f=NormF(y,2,3);
    printf("Norm1_A = %lf\nNorminf_A = %lf\nNormF_A = %lf\nNorm1_B = %lf\nNorminf_B = %lf\nNormF_B = %lf\n",b,a,c,e,d,f);
    return 0;
}

double Norminf(double x[],int n,int m)
{
     
    double a[m];double a1;
    int i,j;
    for(i=0;i<m;i++)
    a[i]=0;
    for(j=0;j<m;j++)
    {
        for(i=n*j;i<n*(j+1);i++)
        a[j]=a[j]+fabs(x[i]);
    }
    a1=a[0];
    for(i=0;i<m;i++)
    {
        if(fabs(a[i])<fabs(a[i+1]))
        a1=fabs(a[i+1]);
    }
    return a1;
}

double NormF(double x[],int n,int m)
{
    double b;
    int i;
    b=0;
    for(i=0;i<n*m;i++)
    b=b+x[i]*x[i]; 
    b=sqrt(b);
    return b;
}

double Norm1(double x[],int n,int m)
{
    double a[n];double a1;
    int i,j;
    for (i=0;i<n;i++)
    a[i]=0;
    for(j=0;j<n;j++)
    {
        for(i=j;i<m*n;i=i+n)
        a[j]=a[j]+fabs(x[i]);
    }
    a1=a[0];
    for(i=0;i<n;i++)
    {
        if(fabs(a[i])<fabs(a[i+1]))
        a1=fabs(a[i+1]);
    }
    return a1;
}
