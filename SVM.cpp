#include<iostream>
#include<cmath>
#include<fstream>
#include "matrix.h"
#include "SVM.h"

void read_csv(double**,double**);
int main()
{
    double** X=new double*[401];
    double** Y=new double*[401];


    for(int i=0;i<401;i++)
    {
        X[i]=new double[3];
        Y[i]=new double[1];
    }
    // CSV Samples...
    read_csv(X,Y);
    X=normalization(X,401,2);
   

    SVM* svm=new SVM(3,-15,15,"gk");
    svm->fit(X,Y,1e6,401,0.000000006,350,0.6);

    //predictions...
    double** y_pred=svm->predict(X,401);
    cout<<"Y        "<<" Y_pred  "<<endl;
    for(int i=300;i<320;i++)
    cout<<Y[i][0]<<"    "<<y_pred[i][0]<<endl;
    cout<<"Accuracy : "<<svm->accuracy()<<endl;
    

   
  

    return 0;
}









void read_csv(double** X,double** Y)
{
string line;
ifstream fin;
int i;



fin.open("Social_Network_Ads.csv");
string temp;

  while(fin && i<401)
  {  
    getline(fin,line);
    

    if(i==0)
    {
        i++;
        continue;
    }
   
    temp="";
    int flag=0;
    int count=0;
    for(int k=0;k<line.length();k++)
    {
        if(flag==1 && line[k]!=',')
        {
            
        temp=temp+line[k];
        continue;
        
        
        }

        
      if(line[k]==',')
      {
          
      if(flag==0)
      {
      flag=1;
      continue;
      }

      if(temp.compare("Male")==0)
      temp="0";
      else if( temp.compare("Female")==0)
      temp="1";
     
     //cout<<temp<<endl;
     
     if(count<3)
       X[i][count]=stoi(temp);
     else if(count>=3)
       Y[i][0]=stoi(temp);
    //cout<<X[i][count]<<endl;  

        temp="";
        
        count++;
        continue;
      }


    }
    Y[i][0]=stoi(temp);
     
    i++;
 
    
  }
  fin.close();  
/* cout<<endl<<"Verify..."<<endl;
 for(int i=0;i<401;i++)
     cout<<X[i][0]<<" "<<X[i][1]<<" "<<X[i][2]<<" ---> "<<Y[i][0]<<endl;;
*/


    
}
