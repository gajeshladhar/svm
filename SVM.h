

using namespace std;

class SVM
{
    double **weights;
    double **bias;
    int features;
    double alpha1;
    double alpha0;
    double** X;
    double** Y;
    int batch_size;
    double loss;
    double lr;
    double C;
    int epochs;
    string kernel;
    double sigma;
    int old_features;
    double** old_input;

    public :
    SVM(int features,double alpha1,double alpha0,string kernel)
    {
        this->features=features;
        this->alpha1=alpha1;
        this->alpha0=alpha0;
        this->kernel=kernel;
        this->old_features=features;

        weights=new double*[features];
        bias=new double*[1];
        bias[0]=new double[1];
       
        for(int i=0;i<features;i++)
        {
            weights[i]=new double[1];
            
        }
       
    }

    double** get_weights()
    {
        return weights;
    }
    double** get_bias()
    {
        return bias;
    }
    double** predict(double** input,int length)
    {
        if(input!=X)
        input=transform_input(input,length);
       
        
        return mat_add(mat_mul(input,weights,length,features,features,1),bias[0][0],length,1);
      /*  for(int i=0;i<length;i++)
        {
            if(result[i][0]>=1)
            result[i][0]=1
            else if(result[i][0]<=-1)
            result[i][0]=0;
        }
        */

    
        
    }
    double get_loss()
    {
        loss=0;
        double** y_pred=predict(X,batch_size);
        double loss0,loss1,loss_theta;

        for(int i=0;i<batch_size;i++)
        {
          
          loss1+=Y[i][0]*(y_pred[i][0]>=1?0:((alpha1*y_pred[i][0])+20));
          loss0+=(1-Y[i][0])*(y_pred[i][0]<=-1?0:((alpha0*(y_pred[i][0])+20)));
        }
        loss_theta=mat_mul(weights,mat_transpose(weights,features,1),features,1,1,features)[0][0];
        loss=C*(loss0+loss1+loss_theta)/1e6;

        return loss;
    }

    void back_prop()
    {
        double** y_pred=predict(X,batch_size);
        for(int i=0;i<batch_size;i++)
        {
            
            if(Y[i][0]==1)
            {
                if(y_pred[i][0]<1)
                {
                    for(int j=0;j<features;j++)
                    weights[j][0]-=lr*(C*alpha1*X[i][j]+2*weights[j][0]);
                    bias[0][0]=bias[0][0]-lr*alpha1*C;
                }
            }
            if(Y[i][0]==0)
            {
                if(y_pred[i][0]>-1)
                {
                    for(int j=0;j<features;j++)
                    weights[j][0]-=lr*(C*alpha0*X[i][j]+2*weights[j][0]);
                    bias[0][0]=bias[0][0]-lr*alpha0*C;
                }
            }
        }
    }


    void fit(double** X,double** Y,double C,int batch_size,double lr,int epochs,double sigma)
    {
        this->X=X;
        this->Y=Y;
        this->C=C;
        this->batch_size=batch_size;
        this->lr=lr;
        this->epochs=epochs;
        this->sigma=sigma;
        this->old_input=X;

        if(kernel.compare("gk")==0)
        {
            weights=new double*[batch_size];
            for(int i=0;i<batch_size;i++)
            weights[i]=new double[1];
            transform_input();
            cout<<"Kernel Trick Activated"<<endl;

        }

        for(int epoch=0;epoch<epochs;epoch++)
        {
            cout<<endl;
            cout<<"Epoch : "<<epoch<<endl;

            back_prop();
            cout<<"Loss :"<<get_loss()<<endl;
            cout<<" W : "<<mat_sum(weights,features,1,0)[0][0]<<endl;

            
        }
    }

    float accuracy()
    {
        double** y_pred=predict(X,batch_size);
        int count=0;
        for(int i=0;i<batch_size;i++)
        {
            if(Y[i][0]==0 && y_pred[i][0]<=-1 || Y[i][0]==1 && y_pred[i][0]>=1)
            count++;
        }
        return count*1.0/batch_size ;
    }


    void transform_input()
    {
        double** temp=new double*[batch_size];
        for(int i=0;i<batch_size;i++)
        temp[i]=new double[batch_size];
        double norm=0;

        for(int i=0;i<batch_size;i++)
        {
            for(int j=0;j<batch_size;j++)
            {
                norm=0;
                for(int k=0;k<features;k++)
                    norm+=(X[i][k]-X[j][k])*(X[i][k]-X[j][k]);
                temp[i][j]=exp(0-norm/(2*sigma*sigma));
            }
        }

        this->X=temp;
        this->features=batch_size;
        
    }

    double** transform_input(double** X,int length)
    {
        double** temp=new double*[length];
        for(int i=0;i<length;i++)
        temp[i]=new double[batch_size];
        double norm=0;

        for(int i=0;i<length;i++)
        {
            for(int j=0;j<batch_size;j++)
            {
                norm=0;
                for(int k=0;k<old_features;k++)
                    norm+=(X[i][k]-this->old_input[j][k])*(X[i][k]-this->old_input[j][k]);
                temp[i][j]=exp(0-norm/(2*sigma*sigma));
            }
        }

       return temp;
        
    }

    
};