# file brnn/methods.R
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

#A package for Sparse L1-norm Regularized Neural Networks
#Author: Yangfan Wang 
#Qingdao, WI, Sep. 2017

summary.snnR=function(object,...)
{
	object
}


print.snnR=function(x,...)
{
  	if(!inherits(x, "snnR")) stop("This function only works for objects of class `snnR'\n");
  
        nlayers=dim(x$nHidden)[2]
        struct=NULL
        for(i in 1:nlayers)
        {
          struct=paste(struct,"-",x$nHidden[i])
        }

  	cat("A sparse neural network with additive effects \n");
  	cat(paste(dim(x$inputwgts)[1],struct,"- 1 with",dim(x$wDNNs)[1], "weights, biases and connection strengths\n",sep=" "));
  	cat("Inputs and output were", ifelse(x$normalize,"","NOT"),"normalized\n",sep=" ");
  	cat("Training finished because ",x$message,"\n");
}


print.snnR_extended=function(x,...)
{
	
        nlayers_add=dim(x$nHidden_add)[2]
        struct_add=NULL
       for(i in 1:nlayers_add)
        {
         struct_add=paste(struct_add,"-",x$nHidden_add[i])
        }
        nlayers_dom=dim(x$nHidden_dom)[2]
        struct_dom=NULL
       for(i in 1:nlayers_dom)
         {
          struct_dom=paste(struct_dom,"-",x$nHidden_dom[i])
         }

        if (!inherits(x, "snnR_extended")) stop("This function only works for objects of class `snnR_extended \n'");
  	cat("A sparse neural network with additive and dominance effects.\n");
  	cat(paste(dim(x$inputwgts_add)[1],struct_add,":",struct_dom,"- 1 with",dim(x$wDNNs_add)[1]+dim(x$wDNNs_dom)[1], "weights, biases and connection strengths\n",sep=" "));
  	cat("Inputs and output were", ifelse(x$normalize,"","NOT"),"normalized\n",sep=" ");
  	cat("Training finished because ",x$message,"\n");
}




predict.snnR=function(object,newdata,...)
{
   y=NULL;

   if(!inherits(object,"snnR")) stop("This function only works for objects of class `snnR \n'");
   
   if (missing(newdata) || is.null(newdata)) 
   {
        
        #y=predictions.nn.C(vecX=as.vector(object$x_normalized),n=object$n,p=object$p,
        #                   theta=object$theta,neurons=object$neurons,cores=1);
        y= Predict(out=object,X=object$X,nHidden=object$nHidden)$yhat;
        
        #predictions in the original scale
        #if(object$normalize)
        #{
        #    y=un_normalize(y,object$y_base,object$y_spread)
        #}
        

   }
   else
   {
      y= Predict(out=object,X=newdata,nHidden=object$nHidden)$yhat;
       #predictions in the original scale
        #if(object$normalize)
        #{
        #    y=un_normalize(y,object$y_base,object$y_spread)
        #}
   }
	
   return(y)
}



predict.snnR_extended=function(object,newdata_add,newdata_dom,...)
{
   y=NULL;

   if(!inherits(object,"snnR_extended")) stop("This function only works for objects of class `snnR_extended \n'");
   
   if (missing(newdata_add) || is.null(newdata_add) || missing(newdata_dom) || is.null(newdata_dom)) 
   {

        #y= Predict(out=object,X=object$X,nHidden=object$nHidden)$yhat;
        y=Predict_extended(out=object,X=object$X,Z=object$Z,nHidden_add=object$nHidden_add,nHidden_dom=object$nHidden_dom)$yhat
        #predictions in the original scale
        #if(object$normalize)
        #{
        #    y=un_normalize(y,object$y_base,object$y_spread)
        #}
        

   }
   else
   {
      y= Predict_extended(out=object,X=newdata_add,Z=newdata_dom,nHidden_add=object$nHidden_add,nHidden_dom=object$nHidden_dom)$yhat;
       #predictions in the original scale
        #if(object$normalize)
        #{
        #    y=un_normalize(y,object$y_base,object$y_spread)
       # }
   }
	
   return(y)
}





















