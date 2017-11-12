#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "util.h"
SEXP lbfgsCR_2(SEXP gX, SEXP sX, SEXP yX, SEXP HX, SEXP nX, SEXP pX)
{
   
    double *s, *y, *g, *H, *d, *ro, *bgt, *cgt, *q, *r;
    int nVars,nSteps;/*,lhs_dims[2];*/
    double temp;
    int i,j;
 
    SEXP answer,answer2, answer3, answer4,answer5, answer6;


    PROTECT(gX=AS_NUMERIC(gX));
    g=NUMERIC_POINTER(gX);
    
    PROTECT(sX=AS_NUMERIC(sX));
    s=NUMERIC_POINTER(sX);
    
    PROTECT(yX=AS_NUMERIC(yX));
    y=NUMERIC_POINTER(yX);

    PROTECT(HX=AS_NUMERIC(HX));
    H=NUMERIC_POINTER(HX);
      
    nVars =INTEGER_VALUE(nX);
    nSteps =INTEGER_VALUE(pX); 

    PROTECT(answer = allocVector(REALSXP,nSteps));
    ro = REAL(answer);
    PROTECT(answer2 = allocVector(REALSXP,nSteps));
    bgt = REAL(answer2);
    PROTECT(answer3 = allocVector(REALSXP,nSteps));
    cgt = REAL(answer3);
    PROTECT(answer4 = allocVector(REALSXP,nVars*(nSteps+1)));
    q = REAL(answer4);
    PROTECT(answer5 = allocVector(REALSXP,nVars*(nSteps+1)));
    r = REAL(answer5);
   PROTECT(answer6 = allocVector(REALSXP,nVars));
    d = REAL(answer6);

    for(i=0;i<nSteps;i++)

    {
        temp = 0;
        for(j=0;j<nVars;j++)
        {
			temp += y[j+nVars*i]*s[j+nVars*i];
        }
        ro[i] = 1/temp;
    }
	
	for(i=0;i<nVars;i++)
	{
		q[i+nVars*nSteps] = g[i];
	}

	for(i=nSteps-1;i>=0;i--)
	{
	
		bgt[i] = 0;
		for(j=0;j<nVars;j++)
		{
			bgt[i] += s[j+nVars*i]*q[j+nVars*(i+1)]; 
		}
		bgt[i] *= ro[i];
		for(j=0;j<nVars;j++)
		{
			q[j+nVars*i]=q[j+nVars*(i+1)]-bgt[i]*y[j+nVars*i];
		}
	}

	for(i=0;i<nVars;i++)
	{
		r[i] = H[0]*q[i];
	}

	for(i=0;i<nSteps;i++)
	{
		cgt[i] = 0;
		for(j=0;j<nVars;j++)
		{
			cgt[i] += y[j+nVars*i]*r[j+nVars*i];
		}
		cgt[i] *= ro[i];
		for(j=0;j<nVars;j++)
		{
			r[j+nVars*(i+1)]=r[j+nVars*i]+s[j+nVars*i]*(bgt[i]-cgt[i]);
		}
	}

	for(i=0;i<nVars;i++)
	{
		d[i]=r[i+nVars*nSteps];
	}
UNPROTECT(10);
return(answer6);
}


