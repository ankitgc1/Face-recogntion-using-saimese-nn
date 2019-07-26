# Face-recogntion-using-siamese-nn

Siamese neural network(based on CNN) widely used for one-shot learning or few-shot learning. This is an example of siamese NN for one-shot face recognition. The model architecture is pretty simple. The input is passed through two parallel CNN, and the absolute difference is taken off last layers. 


### The basic concept of siamese NN 

![alt text](https://github.com/ankitgc1/Face-recogntion-using-siamese-nn/blob/master/saimese_nn.jpeg)


### Model architecture 

*CNN architecture

![alt text](https://github.com/ankitgc1/Face-recogntion-using-siamese-nn/blob/master/parallel_model.png)

*Lambda(parallel CNN's distance) 
 ![alt text](https://github.com/ankitgc1/Face-recogntion-using-siamese-nn/blob/master/lambda.png)
