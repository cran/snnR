# file snnR/snnR.R
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 or 3 of the License
#  (at your option).
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  A copy of the GNU General Public License is available at
#  http://www.r-project.org/Licenses/

#A R package for Sparse  Neural Networks using the Primal-Dual Active-Set Methods for L1 norm Constrained.
#It uses the Primal-Dual Active-Set Methods for L1 norm Constrained.
#(see,D. Krishnan, P. Lin , et al. A Primal-Dual Active-Set Method for Non-Negativity Constrained Total Variation Deblurring Problems[J].
#IEEE Transactions on Image Processing, 2007, 16(11):2766-2777.)
#Author: Yangfan Wang
#Sep. 2016
#Qingdao, China.
#Author: Ping Lin
#Sep. 2016
#Dundee,UK




#' Function to initialize the weights and biases in a neural network.
#' 
#' @param nHidden  architecture of NN ,
#'        
#' @param nvaibles Number of inputs
#'


initw=function(nHidden,nvaibles)
{
  
  npams = nvaibles*nHidden[1,1];
  
  if(length(nHidden[1,])>1)
  {
    for (h in 2:dim(nHidden)[[2]])
    {
      npams = npams+nHidden[1,h-1]*nHidden[1,h];
    }
  }
  npams=npams+nHidden[1, length(nHidden[1,])];
  
  wegt=rnorm(npams);  
  wegt=matrix(wegt,npams,1);
  res<-list(wegt=wegt);
  return(res) 
  
}

#This function will obtain the minimum-norm subgradient of the approximated least squares error.
subgradient<-function (w,X,y,nHidden,lambda,lambda2)
{
  mypenL2<-penalizedLf(w,X,y,nHidden,lambda2) ;
  f= mypenL2[[1]];
  g= mypenL2[[2]];
  f=f+sum(lambda*abs(w));
  pgradient=matrix(0,dim(g)[1],dim(g)[2]);
  pgradient[g < -lambda] = g[g < -lambda] + lambda[g < -lambda];
  pgradient[g > lambda] = g[g > lambda] - lambda[g > lambda];
  nonZero=((w!=0)|(lambda==0));
  pgradient[nonZero] = g[nonZero] + lambda[nonZero]*sign(w[nonZero]);
  return(list(f = f,pgradient = pgradient)) ;

}

# This function will obtain the approximated least squares error  with L1 norm penalty.
penalizedLf<-function (w,X,y,nHidden,lambda2)
{

  Loss <-DNNSLoss(w,X,y,nHidden);
  null=Loss[[1]];
  g=Loss[[2]];
  null=null+sum(lambda2*(w^2));
  g = g + 2*lambda2*w;

  return(list(null = null, g = g)) ;
}


### This function will compute  the approximated least  squares error and it's gradients with respect to parameters.
DNNSLoss <-function (w,X,y,nHidden)
{
  #browser();
  if(any(is.na(w)))
  { f=Inf;
  g=matrix(0,dim(w)[1],dim(w)[2]);
  return;
  }
  nsamples = dim(X)[1];
  nvarables = dim(X)[2];

  NL = length(nHidden)+1;
  if(!(length(nHidden)==0))
  {
    tempnVnH=nvarables*nHidden[[1]];
    inputWgt <-w[1:tempnVnH,1];

    inputWgt=matrix(inputWgt,nvarables,(nHidden[1,1])[[1]]) ;

    setno = nvarables*(nHidden[1,1][[1]]);
    hideWgt=list();
    if(length(nHidden[1,])>1)
    {
      for (h in 2:length(nHidden))
      {
        temnum1= (nHidden[1,h-1])[[1]];
        temnum2= (nHidden[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];

        temp=matrix(temp,nHidden[1,h-1][[1]],nHidden[1,h][[1]]);
        hideWgt[h-1]=list(temp);
        setno = setno+nHidden[1,h-1][[1]]*nHidden[1,h][[1]];
      }
    }
  }
  else
  {
    setno = 0;
  }
  outputWgt =matrix(w[(setno+1):(setno+nHidden[1,length(nHidden)][[1]]),1]);
  f=0;

  if(!(length(nHidden)==0))
  {
    gInput = matrix(0,dim(inputWgt)[1],dim(inputWgt)[2]);
  }
  else
  {
    gInput=matrix();
  }
  gOutput = matrix(0,dim(outputWgt)[1],dim(outputWgt)[2]);

  gHidden=list();
  if(length(nHidden[1,])>1)
  {
    for (h in 1:(length(nHidden)-1))
    {
      gHidden[h]=list(matrix(0,dim(as.matrix(hideWgt[[h]]))[1],dim(as.matrix(hideWgt)[[h]])[2]));
    }
  }
  #  Outputvalue
  ip=list();
  fp=list();

  if(!(length(nHidden)==0))
  {
    ip[1]=list(X%*%inputWgt);
    fp[1]=list(tanh(ip[[1]]));
    ip[[1]][,1]=-Inf;
    fp[[1]][,1]=1;
    if(length(nHidden[1,])>1)
    {
      for (h in 2:(length(nHidden)))
      {   ip[h]=list(fp[[h-1]]%*%hideWgt[[h-1]]);
      fp[h]=list(tanh(ip[[h]]));
      ip[[h]][,1]=-Inf;
      fp[[h]][,1]=1;

      }
    }
    yhat=fp[[length(fp)]]%*%outputWgt;

  }
  else
  {
    yhat=X%*%outputWgt;
  }
  relativeErr = yhat-y;
  f = sum(relativeErr[]^2);
  #Compute gradient
  err = 2*relativeErr;

  #% compute weight gradient
  gOutput=t(fp[[length(fp)]])%*%err;
  if(length(nHidden[1,])>1)
  {
    backprop = 1/cosh(ip[[length(ip)]])^2*(err%*%t(outputWgt));

    gHidden[[length(gHidden)]]=t(fp[[length(fp)-1]])%*%backprop;


    if(length(nHidden[1,])>2)
    {
      for (h in (length(nHidden[1,])-2):1)
      {
        backprop=1/cosh(ip[[h+1]])^2*(backprop%*%t(hideWgt[[h+1]]));
        gHidden[[h]] =  t(fp[[h]])%*%backprop;
      }
    }


    backprop = 1/cosh(ip[[1]])^2*(backprop%*%t(hideWgt[[1]]));
    gInput   =  t(X)%*%backprop;
  }
  else
  {
    if (length(nHidden[1,])==1)
    {

      gInput =  t(X)%*%((1/cosh(ip[[length(ip)]])^2)*(err%*%t(outputWgt)));


    }
  }

  #   gradient  vector
  g=matrix(0,dim(w)[1],dim(w)[2]);
  if(!(length(nHidden)==0))
  {
    g[1:(nvarables*nHidden[1,1][[1]])]=matrix(gInput);
    setno= nvarables*nHidden[1,1][[1]];
    if(length(nHidden[1,])>1)
    {
      for (h in 2:length(nHidden[1,]))
      {
        g[(setno+1):(setno+(nHidden[1,h-1][[1]]*nHidden[1,h][[1]]))] = gHidden[[h-1]];
        setno= setno+nHidden[1,h-1][[1]]*nHidden[1,h][[1]];
      }
    }
  }


  g[(setno+1):(setno+nHidden[1,length(nHidden)][[1]])] = gOutput;

  return(list(f = f, g = g)) ;
}

isLegal<-function(v)
{
  legal =( sum(any(as.logical(Im(v))))==0 & sum(as.numeric(is.nan(matrix(v))))==0 & sum(as.numeric(is.infinite(matrix(v))))==0 );
  return(list(legal=legal)) ;
}

#' Function to normalize a vector or matrix, the resulting all
#'   
#' @examples
#'y_base=min(y)
#'y_spread=max(y) - y_base
#'y_normalized=normalize(y,base=y_base,spread=y_spread)
#'  
normalize=function(x,base,spread)
{
  if(is.matrix(x))
  {
    return(sweep(sweep(x,2,base,"-"),2,2/spread,"*")-1)
  }else{
    return(2*(x-base)/spread-1)
  }
}

#Go back to the original scale
#x=base+0.5*spread*(z+1)
un_normalize=function(z,base,spread)
{
  if(is.matrix(z))
  {
    return(sweep(sweep(z+1,2,0.5*spread,"*"),2,base,"+"))
  }else{
    return(base+0.5*spread*(z+1))
  }	
}


Ew=function(theta)
{
  sum((unlist(theta))^2)
}



# This function has the (L-BFGS) algorithm.
# multiplied by the gradient. Also implemented by c code in this package.
lbfgsF<-function(g,s,y,Hdiag)
{
  if(is.matrix(s)==FALSE)
  {
    s=matrix(s);
  }
  if(is.matrix(y)==FALSE)
  {
    y=matrix(y);
  }

  p=dim(s)[1];
  k=dim(s)[2];
  ro=matrix(0,0,1) ;
  for (i in 1:k )
  {
    ro=rbind(ro,1/(t(y[,i])%*%s[,i]));
  }

  q=matrix(0,p,k+1);
  r=matrix(0,p,k+1);
  al=matrix(0,k,1);
  be=matrix(0,k,1);
  q[,k+1]=g;

  for (i in (k:1))
  {
    al[i]=ro[i]%*%t(s[,i])%*%q[,i+1];
    #q[,i]=q[,i+1]-al[i]%*%y[,i];
    q[,i]=q[,i+1]-al[i]*y[,i];
  }

  r[,1]=Hdiag%*%q[,1];

  for (i in (1:k))
  {
    be[i] = ro[i]%*%t(y[,i])%*%r[,i];
    r[,i+1] = r[,i] + s[,i]%*%matrix((al[i]-be[i]));
  }
  d=r[,k+1];

  return(list(d=d)) ;
}

#This function is  an orthant projection.
orthantProject<-function(w,xi)
{
  w[sign(w) != xi] = 0;
  return(list(w=w)) ;
}

#optimization with L1 norm regularization.
optimization_L1 <-function (w,X,y,nHidden, verbose=FALSE ,lambda,lambda2, optimtol, prgmtol,
                      maxIter, decsuff)
{
  
 if (verbose==FALSE)
  {
    verbose=0
  }

     if (missing(optimtol) || is.null(optimtol)) 
  {
   optimtol = 1e-5;
  }

     if (missing(prgmtol) || is.null(prgmtol)) 
  {
   prgmtol = 1e-9;
  }
 
      if (missing(maxIter) || is.null(maxIter)) 
  {
   maxIter = 1e-5;
  }

     if (missing(decsuff) || is.null(decsuff)) 
  {
   decsuff =1e-4;
  }
 
       if (missing(lambda) || is.null(lambda)) 
  {
        npams=dim(w)[1];
       lambda=matrix(1,npams,1);
  }
        if (missing(lambda2) || is.null(lambda2)) 
  {
        lambda2 = 1e-3;
  }
  
  



  quadraticInit = 1; verbose = 0;corrections=100;Dtype=1;
  nsamples=dim(X)[1];

  tempo=matrix(1,nsamples,1);
  X=cbind(tempo,X);
  nvaibles=dim(X)[2];
  p = length(w);

  pgrad <- subgradient(w,X,y,nHidden,lambda,lambda2);
  f=pgrad[[1]];
  g=pgrad[[2]];
  funEvals = 1;
  optCond = max(abs(g));
  if(optCond < optimtol)
  {
    if (verbose)
    {
      warning("First-order gradient satisfied ",
              call. = F);
    }
    return;
  }

  for (i in 1:maxIter)
  {
    W = (lambda ==0) | (w !=0);
    A = W==0;
    d = matrix(0,p,1);


    if (i == 1)
    {
      d = -g;
      Y=matrix(0,p,0);
      S=matrix(0,p,0);
      sigma = 1;
      wsave=matrix(0,p,0);
      t = min(1,1/sum(abs(g)));
    }
    else
    {
      y2 = g-g_old;
      s = w-w_old;
      correctionsStored = dim(Y)[2];
      if (correctionsStored < corrections)
      {
        Y=cbind(Y,y2);
        S=cbind(S,s);
      }
      else
      {
        Y=cbind(Y[,2:corrections],y2);
        S=cbind(S[,2:corrections],s);
      }

      ys=t(y2)%*%s;
      if (ys > 1e-10)
      {
        sigma = ys/(t(y2)%*%y2);
      }
      jud=dim(S)[2];
      if (jud==1)
      {
        curvSat = colSums(matrix(Y[W,])*matrix(S[W,])) > 1e-10;



        if (length(which(as.numeric(curvSat)==1))>0)
        {
         
          d[W] = lbfgsF(-matrix(g[W]),matrix(S[W,curvSat]),matrix(Y[W,curvSat]),sigma)[[1]];
          #d[W] = lbfgsF(-g[W],S[W,curvSat],Y[W,curvSat],sigma)[[1]];
        }
      }
      else
      {


        tempYS=Y[W,]*S[W,];
        if (is.matrix(tempYS)==FALSE)
        {
          tempYS=matrix(tempYS);
        }
        curvSat = colSums(tempYS) > 1e-10;


        if (length(which(as.numeric(curvSat)==1))>0)
        {
            #R code
            #d[W] = lbfgsF(-g[W],S[W,curvSat],Y[W,curvSat],sigma)[[1]];
            # c code in the following part. 
            if (is.null(dim(S[W,curvSat]))==TRUE)
          {
            d[W] = lbfgsF(-g[W],S[W,curvSat],Y[W,curvSat],sigma)[[1]];
          }
          else
          {
            outd=.Call("lbfgsCR_2",as.double(-g[W]), as.double(S[W,curvSat]),
                       as.double(Y[W,curvSat]), as.double(sigma),as.integer(dim(S[W,curvSat])[1]),
                       as.integer(dim(S[W,curvSat])[2]))
            d[W]=matrix(outd,length(outd),1)

          }
        }
      }

      if (Dtype == 0)
      {
        D=min(1,1/sum(abs(g)));
      }
      else
      {
        D = sigma;
      }
      d[A]=-D*g[A];
      t=1;

    }
    f_old = f;
    g_old = g;
    w_old = w;
  #  obtain orthant
    xi=sign(w);#
    xi[w==0]=sign(-g[w==0]);#
 #   Compute directional derivative

    gtd=t(g)%*%d;
    if (gtd > -prgmtol)
    {
      if (verbose)
      {
        warning("The search direction is below Tol\n",call. = F);
      }
      break;
    }

    if (quadraticInit)
    {
      if (i > 1)
      {
        t = min(1,2*(f-f_prev)/gtd);
      }
      f_prev = f;
    }
    w_new=orthantProject(w+t*d,xi)[[1]];

    pgrad_new <- subgradient(w_new,X,y,nHidden,lambda,lambda2);
    f_new<-pgrad_new[[1]];
    g_new<-pgrad_new[[2]];
    funEvals = funEvals+1;
    while ((f_new > f + decsuff*t(g)%*%(w_new-w))|| !isLegal(f_new)[[1]] )
    {
      t_old = t;
      if( verbose)
      {  warning("Searching...\n",call. = F);
      }

      if (!isLegal(f_new)[[1]])
      {
        if (verbose)
        {
          warning(" Step \n",call. = F);
        }
        t = .5*t;
      }
      else
      {
        wang=c(0, f, gtd, t, f_new, t(g_new)%*%d);
        wang2=matrix(wang,2,3,byrow = TRUE);
        t = polyinterp(wang2)[[1]];
      }

      if (t < t_old*1e-3 )
      {
        if( verbose == 3)
        {
          warning("Interpolated points are too small\n",call. = F);
        } 
        t = t_old*1e-3;
      }
      else
      {
        if (t> t_old*0.6)
        {
          if (verbose == 3)
          {
            warning("Interpolated points are  too large\n",call. = F);
          }
          t = t_old*0.6;
        }
      }


      if (max(abs(t*d)) < prgmtol )
      {
        if (verbose)
        {
          warning("Some wroing in the search direction \n",call. = F);
        }

        t = 0;
        w_new = w;
        f_new = f;
        g_new = g;
        break;
      }


      w_new = orthantProject(w+t*d,xi)[[1]];
      pgrad_new <- subgradient(w_new,X,y,nHidden,lambda,lambda2);
      f_new<-pgrad_new[[1]];
      g_new<-pgrad_new[[2]];
      funEvals = funEvals+1;


    }

    w = w_new;
    f = f_new;
    g = g_new;

    wsave=cbind(wsave,w);
    optCond = max(abs(g));
    if (optCond < optimtol)
    {
      if (verbose)
      {
        warning("First-order optimality below optimtol\n",call. = F);
      }
      break;
    }

    if (max(abs(t*d)) < prgmtol || abs(f-f_old) < prgmtol)
    {
      if (verbose)
      {
        warning(" parameters is below Tol\n",call. = F);
      }
      break;
    }

    if (funEvals >= maxIter)
    {
      if (verbose)
      {
        warning(" maxIter\n",call. = F);
      }
      break;
    }

  }

  return(list(w=w)) ;


}

#   Polynomial interpolation using function and gradient

polyinterp <-function(points)
{

  npois=dim(points)[1];
  order = sum(sum((Im(points[,2:3])==0)))-1;
  xmin = min(points[,1]);
  xmax = max(points[,1]);
  xminbud = xmin;
  xmaxbud = xmax;


  if (npois == 2 && order ==3)
  {
    minVal  = min(points[,1]);
    minPos=which.min(points[,1]);
    notminpois = -minPos+3;

    d1 = points[minPos,3] + points[notminpois,3] - 3*(points[minPos,2]-points[notminpois,2])/(points[minPos,1]-points[notminpois,1]);

    d2 = sqrt(d1^2 - points[minPos,3]*points[notminpois,3]);
    if(is.numeric (d2))
    {

      t = points[notminpois,1] - (points[notminpois,1] - points[minPos,1])*((points[notminpois,3] + d2 - d1)/(points[notminpois,3] - points[minPos,3] + 2*d2));

      minPos = min(max(t,xminbud),xmaxbud);
    }
    else
    {
      minPos = (xmaxbud+xminbud)/2;
    }
    return(list(minPos=minPos));
  }
}

#Inner function obtain predicted values for a NNs with n-hidden-layers.
Predict_mln <-function (w,X,nHidden)
{
  nsamples =dim(X)[1];
  tempo=matrix(1,nsamples,1);
  X=cbind(tempo,X);
  nvarables= dim(X)[2];
  if(!(length(nHidden)==0))
  {

    tempnVnH=nvarables*nHidden[[1]];
    inputwgts <-w[1:tempnVnH,1];

    inputwgts=matrix(inputwgts,nvarables,(nHidden[1,1])[[1]]) ;
    setno = nvarables*(nHidden[1,1][[1]]);


    hidewgts=list();
    if(length(nHidden[1,])>1)
    {
      for (h in 2:length(nHidden))
      {
        temnum1= (nHidden[1,h-1])[[1]];
        temnum2= (nHidden[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];

        temp=matrix(temp,nHidden[1,h-1][[1]],nHidden[1,h][[1]]);   #
        hidewgts[h-1]=list(temp);
        setno = setno+nHidden[1,h-1][[1]]*nHidden[1,h][[1]];
      }
    }
  }
  else
  {
    setno = 0;
  }

  outputwgts =matrix(w[(setno+1):(setno+nHidden[1,length(nHidden)][[1]]),1]);
  ip=list();
  fp=list();

  if(!(length(nHidden)==0))
  {

    ip[1]=list(X%*%inputwgts);
    fp[1]=list(tanh(ip[[1]]));
    ip[[1]][,1]=-Inf;
    fp[[1]][,1]=1;
    if(length(nHidden[1,])>1)
    {
      for (h in 2:(length(nHidden)))
      {   ip[h]=list(fp[[h-1]]%*%hidewgts[[h-1]]);
      fp[h]=list(tanh(ip[[h]]));
      ip[[h]][,1]=-Inf;
      fp[[h]][,1]=1;

      }
    }
    yhat=fp[[length(fp)]]%*%outputwgts;
  }
  else
  {
    yhat=X%*%outputwgts;
  }


  #browser();
  return(list(yhat=yhat,inputwgts=inputwgts,outputwgts=outputwgts,hidewgts=hidewgts));

}


#This function obtain predicted values a sparse NN for additive effects.
Predict <-function (out,X,nHidden)
{
 # browser();
  w=out$wDNNs;
  nsamples =dim(X)[1];
  tempo=matrix(1,nsamples,1);
  X=cbind(tempo,X);
  nvarables= dim(X)[2];
  if(!(length(nHidden)==0))
  {

    tempnVnH=nvarables*nHidden[[1]];
    inputwgts <-w[1:tempnVnH,1];

    inputwgts=matrix(inputwgts,nvarables,(nHidden[1,1])[[1]]) ;
    setno = nvarables*(nHidden[1,1][[1]]);


    hidewgts=list();
    if(length(nHidden[1,])>1)
    {
      for (h in 2:length(nHidden))
      {
        temnum1= (nHidden[1,h-1])[[1]];
        temnum2= (nHidden[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];

        temp=matrix(temp,nHidden[1,h-1][[1]],nHidden[1,h][[1]]);   #
        hidewgts[h-1]=list(temp);
        setno = setno+nHidden[1,h-1][[1]]*nHidden[1,h][[1]];
      }
    }
  }
  else
  {
    setno = 0;
  }

  outputwgts =matrix(w[(setno+1):(setno+nHidden[1,length(nHidden)][[1]]),1]);
  ip=list();
  fp=list();

  if(!(length(nHidden)==0))
  {

    ip[1]=list(X%*%inputwgts);
    fp[1]=list(tanh(ip[[1]]));
    ip[[1]][,1]=-Inf;
    fp[[1]][,1]=1;
    if(length(nHidden[1,])>1)
    {
      for (h in 2:(length(nHidden)))
      {   ip[h]=list(fp[[h-1]]%*%hidewgts[[h-1]]);
      fp[h]=list(tanh(ip[[h]]));
      ip[[h]][,1]=-Inf;
      fp[[h]][,1]=1;

      }
    }
    yhat=fp[[length(fp)]]%*%outputwgts;
  }
  else
  {
    yhat=X%*%outputwgts;
  }
  #browser()
  if(!is.null(out$y_base)&&!is.null(out$y_spread))
  {
    yhat=un_normalize(yhat,out$y_base,out$y_spread)
  }
  #browser();
  return(list(yhat=yhat,inputwgts=inputwgts,outputwgts=outputwgts,hidewgts=hidewgts));

}




##################################
# 'rmse': Root Mean Square Error #

rmse <-function(sim, obs, ...) UseMethod("rmse")

rmse.default <- function (sim, obs, na.rm=TRUE, ...) {

  if ( is.na(match(class(sim), c("integer", "numeric", "ts"))) |
          is.na(match(class(obs), c("integer", "numeric", "ts")))
     ) stop("Invalid argument type: 'sim' & 'obs' have to be of class: c('integer', 'numeric', 'ts')")    
      
  if ( length(obs) != length(sim) ) 
	 stop("Invalid argument: 'sim' & 'obs' doesn't have the same length !")     
	 
  rmse <- sqrt( mean( (sim - obs)^2, na.rm = na.rm) )
           
  return(rmse)
     
} # 'rmse.default' end
  

rmse.matrix <- function (sim, obs, na.rm=TRUE, ...) {

   # Checking that 'sim' and 'obs' have the same dimensions
   if ( all.equal(dim(sim), dim(obs)) != TRUE )
    stop( paste("Invalid argument: dim(sim) != dim(obs) ( [", 
          paste(dim(sim), collapse=" "), "] != [", 
          paste(dim(obs), collapse=" "), "] )", sep="") )

   rmse <- sqrt( colMeans( (sim - obs)^2, na.rm = na.rm, ...) )          
           
   return(rmse)
     
} # 'rmse.matrix' end


rmse.data.frame <- function (sim, obs, na.rm=TRUE, ...) {

   rmse <- sqrt( colMeans( (sim - obs)^2, na.rm = na.rm, ...) )          
           
   return(rmse)
     
} # 'rmse.data.frame' end


#This function will obtain predicted values of a sparse NN for additive and dominance effects.
Predict_extended <-function (out,X,Z,nHidden_add,nHidden_dom)
{
   #browser();
  w=out$wDNNs_add;
  wz=out$wDNNs_dom
  nsamples =dim(X)[1];
  tempo=matrix(1,nsamples,1);
  X=cbind(tempo,X);
  nvarables= dim(X)[2];
  if(!(length(nHidden_add)==0))
  {

    tempnVnH=nvarables*nHidden_add[[1]];
    inputwgts <-w[1:tempnVnH,1];

    inputwgts=matrix(inputwgts,nvarables,(nHidden_add[1,1])[[1]]) ;
    setno = nvarables*(nHidden_add[1,1][[1]]);


    hidewgts=list();
    if(length(nHidden_add[1,])>1)
    {
      for (h in 2:length(nHidden_add))
      {
        temnum1= (nHidden_add[1,h-1])[[1]];
        temnum2= (nHidden_add[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];

        temp=matrix(temp,nHidden_add[1,h-1][[1]],nHidden_add[1,h][[1]]);   #
        hidewgts[h-1]=list(temp);
        setno = setno+nHidden_add[1,h-1][[1]]*nHidden_add[1,h][[1]];
      }
    }
  }
  else
  {
    setno = 0;
  }

  outputwgts =matrix(w[(setno+1):(setno+nHidden_add[1,length(nHidden_add)][[1]]),1]);
  ip=list();
  fp=list();

  if(!(length(nHidden_add)==0))
  {

    ip[1]=list(X%*%inputwgts);
    fp[1]=list(tanh(ip[[1]]));
    ip[[1]][,1]=-Inf;
    fp[[1]][,1]=1;
    if(length(nHidden_add[1,])>1)
    {
      for (h in 2:(length(nHidden_add)))
      {   ip[h]=list(fp[[h-1]]%*%hidewgts[[h-1]]);
      fp[h]=list(tanh(ip[[h]]));
      ip[[h]][,1]=-Inf;
      fp[[h]][,1]=1;

      }
    }
    yhat=fp[[length(fp)]]%*%outputwgts;
  }
  else
  {
    yhat=X%*%outputwgts;
  }

  nsamplesz =dim(Z)[1];
  tempoz=matrix(1,nsamplesz,1);
  Z=cbind(tempoz,Z);
  nvarablesz= dim(Z)[2];
  if(!(length(nHidden_dom)==0))
  {

    tempnVnHz=nvarablesz*nHidden_dom[[1]];
    inputwgtsz <-wz[1:tempnVnHz,1];

    inputwgtsz=matrix(inputwgtsz,nvarablesz,(nHidden_dom[1,1])[[1]]) ;
    setnoz = nvarablesz*(nHidden_dom[1,1][[1]]);


    hidewgtsz=list();
    if(length(nHidden_dom[1,])>1)
    {
      for (h in 2:length(nHidden_dom))
      {
        temnum1z= (nHidden_dom[1,h-1])[[1]];
        temnum2z= (nHidden_dom[1,h])[[1]];
        temnumz=temnum1z*temnum2z;
        tempz<-wz[(setnoz+1):(setnoz+temnumz),1];

        tempz=matrix(tempz,nHidden_dom[1,h-1][[1]],nHidden_dom[1,h][[1]]);   #
        hidewgtsz[h-1]=list(tempz);
        setnoz = setnoz+nHidden_dom[1,h-1][[1]]*nHidden_dom[1,h][[1]];
      }
    }
  }
  else
  {
    setnoz = 0;
  }

  outputwgtsz =matrix(wz[(setnoz+1):(setnoz+nHidden_dom[1,length(nHidden_dom)][[1]]),1]);
  ipz=list();
  fpz=list();

  if(!(length(nHidden_dom)==0))
  {

    ipz[1]=list(Z%*%inputwgtsz);
    fpz[1]=list(tanh(ipz[[1]]));
    ipz[[1]][,1]=-Inf;
    fpz[[1]][,1]=1;
    if(length(nHidden_dom[1,])>1)
    {
      for (h in 2:(length(nHidden_dom)))
      {   ipz[h]=list(fpz[[h-1]]%*%hidewgtsz[[h-1]]);
      fpz[h]=list(tanh(ipz[[h]]));
      ipz[[h]][,1]=-Inf;
      fpz[[h]][,1]=1;

      }
    }
    yhatz=fpz[[length(fpz)]]%*%outputwgtsz;
  }
  else
  {
    yhatz=Z%*%outputwgtsz;
  }

  yhat=yhat+yhatz;

  if(!is.null(out$y_base)&&!is.null(out$y_spread))
  {
    yhat=un_normalize(yhat,out$y_base,out$y_spread)
  }
  return(list(yhat=yhat,inputwgts_add=inputwgts,outputwgts_add=outputwgts,hidewgts_add=hidewgts,inputwgts_dom=inputwgtsz,outputwgts_dom=outputwgtsz,hidewgts_dom=hidewgtsz));

}


snnR<-function (x,y,nHidden,normalize=FALSE, verbose=TRUE,optimtol = 1e-5, prgmtol = 1e-9, iteramax = 100, decsuff =1e-4,lambda)
{
  #Checking that the imputs are ok
  if(!is.vector(y)) stop("y must be a vector\n")
  if(!is.matrix(x)) stop("x must be a matrix\n")
  if(!is.matrix(nHidden)) stop("nHidden must be a matrix with 1 times h.\n")
  if(dim(nHidden)[1]!=1) stop("nHidden must be a matrix with 1 times h.\n")
  
  if(iteramax<5) stop("iteramax must be bigger than 5\n")
  y=matrix(y,length(y),1);
  y_base=NULL;
  y_spread=NULL;
  X=x;
  nHidden=round(nHidden);
  if(normalize)
  {
    y_base=min(y)
    y_spread=max(y) - y_base
    y_normalized=normalize(y,base=y_base,spread=y_spread)
    y=y_normalized;

  }

   if (missing(lambda) || is.null(lambda)) 
  {
    lambda=1
  }
  

  if(iteramax==1)
  {
    iteramax=iteramax+1;
  }
  nsamples=dim(X)[1];
  nvaibles=dim(X)[2];
  nvaibles = nvaibles+1;
  wegt=initw(nHidden,nvaibles)$wegt;
  wDNNs =wegt;#
  npams=dim(wegt)[1];
  ## set lambda
  lambda =lambda;
  lambda=matrix(lambda,npams,1);
  lambda2 = 1e-3;
  MLPPredict=Predict_mln(wDNNs,X,nHidden);
  yhat = MLPPredict[[1]];
  relativeErr = yhat-y;
  fsqure = sum(relativeErr[]^2);
  Mse=fsqure/nsamples;
  difmse=0.00001
  if(!is.null(y_base)&&!is.null(y_spread))
  {
    yhat2=un_normalize(yhat,y_base,y_spread)
    y2=un_normalize(y,y_base,y_spread)
    rMse=rmse(yhat2,y2)
      if (verbose==TRUE)
      {
      cat("Training sparse neural networks ...,", "The root mean squared error :",rMse,". \n")
      }
  }
  else
  {   if (verbose==TRUE)
     {
      cat("Training sparse  neural networks ...,", "The mean squared error :",Mse,". \n");
     }
   }
  i=1;
  chongfu=1;
  timecout=NULL;
  timerec=0;
  rmse=NULL;
   while(i<=iteramax) 
  {
     Msetol=Mse;
     wegt_old = wDNNs;
     kk= optimization_L1(w=wDNNs,X,y,nHidden, 0 ,lambda,lambda2, optimtol,prgmtol,500,decsuff);
     wDNNs =kk[[1]];
     nwDNNS=dim(wDNNs)[1];
     non=wDNNs[wDNNs!=0];
     nonwD=length(non);
    s=i;
    if(nonwD<5&&nvaibles>10)
    {
      wegt=initw(nHidden,nvaibles)$wegt;
      wDNNs =wegt;#%
      s=1;
      wegt_old = wDNNs;
      kk= optimization_L1(w=wDNNs,X,y,nHidden, 0 ,lambda,lambda2, optimtol,prgmtol,500,decsuff);
      wDNNs =kk[[1]];

    }
    i=s;
    MLPPredict=Predict_mln(wDNNs,X,nHidden);
    yhat = MLPPredict[[1]];
    relativeErr = yhat-y;
    fsqure = sum(relativeErr[]^2);
    Mse=fsqure/nsamples;
    if(!is.null(y_base)&&!is.null(y_spread))
    {
      yhat2=un_normalize(yhat,y_base,y_spread)
      y2=un_normalize(y,y_base,y_spread)
      rMse=rmse(yhat2,y2)
      if (verbose==TRUE)
      {
      cat("Iteration:",i,", Number of parameters (weights and biases) to estimate:",nwDNNS,", Number of non-zero parameters:",nonwD,", The root mean squared error :",rMse,". \n")
       }
     
    }
    else
    {
      if (verbose==TRUE)
      {
      cat("Iteration:",i,", Number of parameters (weights and biases) to estimate:",nwDNNS,", Number of non-zero parameters:",nonwD,", The mean squared error :",Mse,". \n")
      }
    }
    i=i+1;
    if (norm(matrix(wegt_old-wDNNs),'i') < 1e-4)
    {
      break;
    }


   }

  mess1=norm(matrix(wegt_old-wDNNs),'i') ;

  message=paste("Iteration:",as.character(i),",Number of parameters (weights and biases) to estimate:"
                ,as.character(nwDNNS), ", Number of non-zero parameters:",as.character(nonwD),
                ", L1 norm of the difference of parameters in the last two interation:",as.character(mess1),".\n" );

 
  w=wDNNs;
  nsamples =dim(X)[1];
  nvarables= dim(X)[2]+1;
  if(!(length(nHidden)==0))
  {

    tempnVnH=nvarables*nHidden[[1]];
    inputwgts <-w[1:tempnVnH,1];

    inputwgts=matrix(inputwgts,nvarables,(nHidden[1,1])[[1]]) ;
    setno = nvarables*(nHidden[1,1][[1]]);


    hidewgts=list();
    if(length(nHidden[1,])>1)
    {
      for (h in 2:length(nHidden))
      {
        temnum1= (nHidden[1,h-1])[[1]];
        temnum2= (nHidden[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];

        temp=matrix(temp,nHidden[1,h-1][[1]],nHidden[1,h][[1]]);   
        hidewgts[h-1]=list(temp);
        setno = setno+nHidden[1,h-1][[1]]*nHidden[1,h][[1]];
      }
    }
  }
  else
  {
    setno = 0;
  }

  outputwgts =matrix(w[(setno+1):(setno+nHidden[1,length(nHidden)][[1]]),1]);

  MLPPredict=Predict_mln(wDNNs,X,nHidden);
  yhat = MLPPredict[[1]];
  relativeErr = yhat-y;
  fsqure = sum(relativeErr[]^2);
  Mse=fsqure/nsamples;
  relativeErr_base=min(relativeErr)
  relativeErr_spread=max(relativeErr) - relativeErr_base
  relativeErr_normalized=normalize(relativeErr,base=relativeErr_base,spread=relativeErr_spread)
  relativeErr_normalized_fsqure = sum(relativeErr_normalized[]^2);

  beta_1=sum(relativeErr_normalized_fsqure);

  wDNNs_base=min(wDNNs)
  wDNNs_spread=max(wDNNs)-wDNNs_base
  wDNNs_normalized=normalize(wDNNs,base=wDNNs_base,spread = wDNNs_spread)
  alpha_1=Ew(wDNNs_normalized)
  out=list(wDNNs=wDNNs,X=X,y=y,nHidden=nHidden,normalize=normalize,inputwgts=inputwgts,outputwgts=outputwgts,hidewgts=hidewgts,Mse=Mse,message=message,y_base=y_base
,y_spread=y_spread,alpha=alpha_1,beta=beta_1)
  class(out) <- "snnR"
  return(out);


}



snnR_extended <-function (x,y,z,nHidden_add,nHidden_dom,normalize=TRUE, verbose=TRUE, optimtol = 1e-5, prgmtol = 1e-9, iteramax = 20, decsuff =1e-4,lambda)
{

  if(!is.vector(y)) stop("y must be a vector\n")
  if(!is.matrix(x)) stop("x must be a matrix\n")
  if(!is.matrix(z)) stop("z must be a matrix\n")
  if(!is.matrix(nHidden_add)) stop("nHidden_add must be a matrix with 1 times h.\n")
  if(!is.matrix(nHidden_dom)) stop("nHidden_dom must be a matrix with 1 times h.\n")
  if(dim(nHidden_add)[1]!=1) stop("nHidden must be a matrix with 1 times h.\n")
  if(dim(nHidden_dom)[1]!=1) stop("nHidden must be a matrix with 1 times h.\n")
   nHidden_add=round(nHidden_add)
   nHidden_dom=round(nHidden_dom)
 
  if(iteramax<5) stop("iteramax must be bigger than 5\n")
   iteramax=round(iteramax)
  y=as.matrix(y)
  X=x
  Z=z
  if(normalize)
  {
    y_base=min(y)
    y_spread=max(y) - y_base
    y_normalized=normalize(y,base=y_base,spread=y_spread)
    y=y_normalized;

  }

  if (missing(lambda) || is.null(lambda)) 
  {
    lambda=1
  }
  lambda_org=lambda
  
  if(iteramax==1)
  {
    iteramax=iteramax+1;
  }
  nsamples=dim(X)[1];
  nvaibles=dim(X)[2];
  nvaibles = nvaibles+1;

  nsamples_z=dim(Z)[1];
  nvaibles_z=dim(Z)[2];
  nvaibles_z = nvaibles_z+1;
  difmse=0.00001
  wegt=initw(nHidden_add,nvaibles)$wegt;
  wDNNs =wegt;#
  npams=dim(wegt)[1];
  lambda =lambda;
  lambda=matrix(lambda,npams,1);
  lambda2 = 1e-3;
  MLPPredict=Predict_mln(wDNNs,X,nHidden_add);
  yhat = MLPPredict[[1]];
  wegt_z=initw(nHidden_dom,nvaibles_z)$wegt;
  wDNNs_z =wegt_z;#
  npams_z=dim(wegt_z)[1];
  
  lambda_z =lambda_org;
  lambda_z=matrix(lambda_z,npams_z,1);

  lambda2_z = 1e-3;
  MLPPredict_z=Predict_mln(wDNNs_z,Z,nHidden_dom);
  yhat_z = MLPPredict_z[[1]];
  relativeErr = y-yhat-yhat_z;
  fsqure = sum(relativeErr[]^2);
  Mse=fsqure/(nsamples+nsamples_z);
  if(!is.null(y_base)&&!is.null(y_spread))
  {
    yhat2=un_normalize(yhat,y_base,y_spread)
    yhat_z2=un_normalize(yhat_z,y_base,y_spread)
    y2=un_normalize(y,y_base,y_spread)
    rMse=rmse(yhat2+yhat_z2,y2)
     if (verbose==TRUE)
      {
     cat("Training sparse neural networks ...,", "The root  mean squared error :",rMse,". \n")
      }
  }
  else
  {  if (verbose==TRUE)
      {
     cat("Training sparse neural networks ...,", "The   mean squared error :",Mse,". \n")
      }
  }

  i=1;

  while(i<=iteramax) 
  {
    Msetol=Mse;
    wegt_old = wDNNs;
    kk= optimization_L1(w=wDNNs,X,y,nHidden_add, 0 ,lambda,lambda2, optimtol,prgmtol,500,decsuff);
    wegt_old_z = wDNNs_z;
    kk_z= optimization_L1(w=wDNNs_z,Z,y,nHidden_dom, 0 ,lambda_z,lambda2_z, optimtol,prgmtol,500,decsuff);
    wDNNs =kk[[1]];
    nwDNNS=dim(wDNNs)[1];
    non=wDNNs[wDNNs!=0];
    nonwD=length(non);
    wDNNs_z =kk_z[[1]];
    nwDNNS_z=dim(wDNNs_z)[1];
    non_z=wDNNs_z[wDNNs_z!=0];
    nonwD_z=length(non_z);
    s=i;
    if(nonwD<5&&nvaibles>10)
    {
      s=1;
      wegt=initw(nHidden_add,nvaibles)$wegt;
      wDNNs =wegt;#
      wegt_old = wDNNs;
      kk= optimization_L1(w=wDNNs,X,y,nHidden_add, 0 ,lambda,lambda2, optimtol,prgmtol,500,decsuff);
      wDNNs =kk[[1]];
    }

    if( nvaibles_z>10&&nonwD_z<5)
    {
      s=1;
      wegt_z=initw(nHidden_dom,nvaibles_z)$wegt;
      wDNNs_z =wegt_z;
      wegt_old_z = wDNNs_z;
      kk_z= optimization_L1(w=wDNNs_z,Z,y,nHidden_dom, 0 ,lambda_z,lambda2_z, optimtol,prgmtol,500,decsuff);
      wDNNs_z =kk_z[[1]];
    }
    i=s;
    s=i;
    MLPPredict=Predict_mln(wDNNs,X,nHidden_add);
    yhat = MLPPredict[[1]];
    MLPPredict_z=Predict_mln(wDNNs_z,Z,nHidden_dom)
    yhat_z = MLPPredict_z[[1]]
    relativeErr = y-yhat-yhat_z
    fsqure = sum(relativeErr[]^2)
    Mse=fsqure/(nsamples+nsamples_z)
    if(!is.null(y_base)&&!is.null(y_spread))
    {
      yhat2=un_normalize(yhat,y_base,y_spread)
      yhat_z2=un_normalize(yhat_z,y_base,y_spread)
      y2=un_normalize(y,y_base,y_spread)
      rMse=rmse(yhat2+yhat_z2,y2)
      if (verbose==TRUE)
      {
      cat("Iteration:",i,", Number of parameters (weights and biases) to estimate:",nwDNNS,", Number of non-zero parameters:",nonwD,", The root mean squared error :",rMse,". \n")
      }
    }
    else
    {
       if (verbose==TRUE)
      {
        cat("Iteration:",i,", Number of parameters (weights and biases) to estimate:",nwDNNS,", Number of non-zero parameters:",nonwD,", The  mean squared error :",Mse,". \n")
      }
    }
    i=i+1;
    if (norm(matrix(wegt_old-wDNNs),'i') < 1e-4 && norm(matrix(wegt_old_z-wDNNs_z),'i') < 1e-4)
    {
      break;
    }
    }
  mess1=norm(matrix(wegt_old-wDNNs),'i') ;
  mess1_z=norm(matrix(wegt_old_z-wDNNs_z),'i') ;
  message=paste("Iteration:",as.character(i),",Number of parameters (weights and biases) to estimate:"
                ,as.character(nwDNNS+nwDNNS_z), ", Number of non-zero parameters:",as.character(nonwD+nonwD_z),
                ", L1 norm of the difference of parameters in the last two interation:",as.character(mess1+mess1_z),".\n" );

  w=wDNNs;
  nsamples =dim(X)[1];
  nvarables= dim(X)[2]+1;
  if(!(length(nHidden_add)==0))
  {

    tempnVnH=nvarables*nHidden_add[[1]];
    inputwgts <-w[1:tempnVnH,1];

    inputwgts=matrix(inputwgts,nvarables,(nHidden_add[1,1])[[1]]) ;
    setno = nvarables*(nHidden_add[1,1][[1]]);


    hidewgts=list();
    if(length(nHidden_add[1,])>1)
    {
      for (h in 2:length(nHidden_add))
      {
        temnum1= (nHidden_add[1,h-1])[[1]];
        temnum2= (nHidden_add[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];

        temp=matrix(temp,nHidden_add[1,h-1][[1]],nHidden_add[1,h][[1]]);   #
        hidewgts[h-1]=list(temp);
        setno = setno+nHidden_add[1,h-1][[1]]*nHidden_add[1,h][[1]];
      }
    }
  }
  else
  {
    setno = 0;
  }

  outputwgts =matrix(w[(setno+1):(setno+nHidden_add[1,length(nHidden_add)][[1]]),1]);


  w_z=wDNNs_z;
  nsamples_z =dim(Z)[1];
  nvarables_z= dim(Z)[2]+1;
  if(!(length(nHidden_dom)==0))
  {

    tempnVnH_z=nvarables_z*nHidden_dom[[1]];
    inputwgts_z <-w_z[1:tempnVnH_z,1];
    inputwgts_z=matrix(inputwgts_z,nvarables_z,(nHidden_dom[1,1])[[1]]) ;
    setno_z = nvarables_z*(nHidden_dom[1,1][[1]]);
    hidewgts_z=list();
    if(length(nHidden_dom[1,])>1)
    {
      for (h in 2:length(nHidden_dom))
      {
        temnum1_z= (nHidden_dom[1,h-1])[[1]];
        temnum2_z= (nHidden_dom[1,h])[[1]];
        temnum_z=temnum1_z*temnum2_z;
        temp_z<-w_z[(setno_z+1):(setno_z+temnum_z),1];

        temp_z=matrix(temp_z,nHidden_dom[1,h-1][[1]],nHidden_dom[1,h][[1]]);   #
        hidewgts_z[h-1]=list(temp_z);
        setno_z = setno_z+nHidden_dom[1,h-1][[1]]*nHidden_dom[1,h][[1]];
      }
    }
  }
  else
  {
    setno_z = 0;
  }
  outputwgts_z =matrix(w_z[(setno_z+1):(setno_z+nHidden_dom[1,length(nHidden_dom)][[1]]),1]);

  MLPPredict=Predict_mln(wDNNs,X,nHidden_add);
  yhat = MLPPredict[[1]];
  ## z
  MLPPredict_z=Predict_mln(wDNNs_z,Z,nHidden_dom);
  yhat_z = MLPPredict_z[[1]];
  relativeErr = yhat_z+yhat-y;
  fsqure = sum(relativeErr[]^2);
  Mse=fsqure/nsamples;
  out= list(wDNNs_add=wDNNs,wDNNs_dom=wDNNs_z,X=X,y=y,Z=Z,nHidden_add=nHidden_add,nHidden_dom=nHidden_dom,normalize=normalize,inputwgts_add=inputwgts,outputwgts_add=outputwgts,hidewgts_add=hidewgts, inputwgts_dom=inputwgts_z,outputwgts_dom=outputwgts_z,hidewgts_dom=hidewgts_z,Mse=Mse,message=message,y_base=y_base,y_spread=y_spread)
  class(out) <- "snnR_extended"
  return(out);

}




#' Function to show  the optimal structures of layer architectures by passing
#' the parameters of sdnn into the function plotnet of NeuralNetTools (An R
#'package at the CRAN site).
#' 

write.NeuralNetTools <-function (w,nHidden,x,y)
{
  
  if(!is.vector(y)) stop("y must be a vector\n")
  if(!is.matrix(x)) stop("x must be a matrix\n")
  if(!is.matrix(w)) stop("w must be a matrix\n")
  
  y=matrix(y,length(y),1);
  #browser();
  #X=as.matrix(x);
  #y=as.matrix(y)
  X=x;
  nsamples = dim(X)[1];
  tempo=matrix(1,nsamples,1);
  X=cbind(tempo,X);
  nvarables = dim(X)[2];  
  NL = length(nHidden)+1;
  if(!(length(nHidden)==0))  
  {
    tempnVnH=nvarables*nHidden[[1]];
    inputWgt <-w[1:tempnVnH,1];
    
    inputWgt=matrix(inputWgt,nvarables,(nHidden[1,1])[[1]]) ;  
    
    setno = nvarables*(nHidden[1,1][[1]]);
    hideWgt=list();
    hideWgtplot=list();
    if(length(nHidden[1,])>1) 
    {    
      for (h in 2:length(nHidden)) 
      {
        temnum1= (nHidden[1,h-1])[[1]];
        temnum2= (nHidden[1,h])[[1]];
        temnum=temnum1*temnum2;
        temp<-w[(setno+1):(setno+temnum),1];
        
        temp=matrix(temp,nHidden[1,h-1][[1]],nHidden[1,h][[1]]); 
        tempplot=rbind(matrix(0,1,dim(temp)[2]),temp);
        hideWgt[h-1]=list(temp);
        hideWgtplot[h-1]=list(tempplot);
        setno = setno+nHidden[1,h-1][[1]]*nHidden[1,h][[1]]; 
      }
    }  
  }
  else
  {
    setno = 0;
  }
  outputWgt =matrix(w[(setno+1):(setno+nHidden[1,length(nHidden)][[1]]),1]); 
  outputWgtplot=rbind(0,outputWgt);
  w_re1=matrix(inputWgt,byrow = TRUE);
  if (length(nHidden[1,])>1) 
  {    
    w_re2=matrix(hideWgtplot[[1]],byrow =TRUE);
    for (h in 2:(length(nHidden)-1)) 
    {
      w_re2=rbind(w_re2,matrix(hideWgtplot[[h]],byrow = TRUE) );
    }
  }
  w_re3=matrix(outputWgtplot,byrow = TRUE);
  w_reall=rbind(w_re1,w_re2,w_re3);
  w_reall=as.numeric(w_reall);
  w_reall[w_reall==0]<-NA;
  ninputs=dim(X)[2]-1;
  noutputs=dim(y)[2];
  structure=c(ninputs,nHidden,noutputs);
  
  return(list(w_re=w_reall,structure=structure)) ;
  
}





#' Function to create simulation data of nonlinear function or nonlinear classification. 
#' @examples
#'
#' nsamples = 200
#' nvaibles = 1
#' Xydata=SimData("Nonlinearregress",nsamples,nvaibles)# simulation data of nonlinear function
#' X=Xydata[[1]];
#' y=Xydata[[2]];
#' 
#'Classdata=SimData("Nonlinearclassification",nsamples,nvaibles,2)#  simulation data of nonlinear classification
#' @export
#'

SimData <-function ( type = c("Nonlinearclassification", "Nonlinearregress"),nsamples,nvaibles,nClasses)   
{
  # check g2 function argument
    if (length(type) == 2){
        type <- "Nonlinearregress"
    } else if (!((type == "Nonlinearclassification")|(type == "Nonlinearregress"))){
        stop("type argument needs to be Nonlinearclassification or Nonlinearregress")
    } 
  m=nsamples;
  n=nvaibles;
  X = 2*matrix(runif(m*n),m,n)-1;
  if(n==1)
  {
    X=matrix(X);
  }
  
  if(type=="Nonlinearclassification")
  {
    nExamplePoints = 5;
    nClasses = 2;
     examplePoints=rnorm(nClasses*nExamplePoints,nvaibles)
      y=matrix(0,nrow=nsamples,ncol=1)

        for (i in 1:nsamples)
          {
            
             tempw=matrix(X[i,],nClasses*nExamplePoints,1,byrow =TRUE); 
             dists = rowSums((tempw - examplePoints)^2); 
             dists=matrix(dists,nClasses*nExamplePoints,1);
             minVal=min(dists);
              minInd=which(dists==minVal);
            y[i,1]=sign(minInd%%nClasses-0.5)   ;
        }
 

  }
  
  if(type=="Nonlinearregress")
  {
    X = 10*matrix(runif(m*n),m,n)-5; 
    var = .1;                                
    if(n==1)
    {
      X=matrix(X);
    }
    nExampos = 20;
    exampos = 10*matrix(runif(nExampos*nvaibles), nExampos,nvaibles)-5; 
    examtag = 10*matrix(runif(nExampos*1),nExampos,1)-5;  
    y = matrix(rep(0,nsamples))  
    
    for (i in 1:nsamples)
    {
      
      tempw=matrix(X[i,],nExampos,length(X[i,]),byrow =TRUE); 
      dists = rowSums(abs(tempw - exampos)); 
      dists=matrix(dists,nExampos,1);
      lik = (1/sqrt(2*pi))*exp(-dists/(2*var));  
      lik = lik/colSums(lik);
      lik=matrix(lik,nExampos,1);
      y[i,1] = t(lik)%*%examtag + runif(1)/15; 
      
    }
    
   
  }
  res<-list(X=X,y=y);
  return (res)
  
}

