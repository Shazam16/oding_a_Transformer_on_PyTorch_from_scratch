This project implements a multilayer perceptron from scratch in pure C. It solves the XOR problem using one hidden layer (4 neurons), sigmoid activation, and mean squared error loss. Forward/backward propagation and stochastic gradient descent update weights manually-no frameworks, no external libraries. The code demonstrates low-level memory management, the chain rule, and portable ML suitable for embedded systems. Compile with any C compiler.

Network Architecture (Multilayer Perceptron)

Input layer      Hidden layer (4 neurons)    Output layer
   (2 nodes)            (sigmoid)              (1 node)
     o ──────────────────────────────────────────► o
     │    ┌───┐   ┌───┐   ┌───┐   ┌───┐
     └───►│   │──►│   │──►│   │──►│   │──► prediction
          └───┘   └───┘   └───┘   └───┘
    W1, b1               W2, b2

    Configuration:

Input size = 2

Hidden size = 4

Output size = 1

Activation = sigmoid (all layers)

Loss = Mean Squared Error (MSE)

XOR Problem – The Classic Non‑Linear Benchmark

Input A	Input B	XOR Output
      0     0    0
 	    0     1    1 
      1     0    1
      1     1    0

      Visualisation (ASCII):

            (0,1)● 1           (1,1)● 0
      
      (0,0)● 0           (1,0)● 1

      Red circles (0) and green circles (1) cannot be separated by a single line – requires hidden layer

      Neural Network Data Structure (C)

      typedef struct {
    int input_size, hidden_size, output_size;

    double *W1, *b1;   // input->hidden weights & biases
    double *W2, *b2;   // hidden->output weights & biases

    // Cache arrays for forward/backward pass
    double *z1, *a1, *z2, *a2;
} NeuralNetwork;

Memory layout:

W1: input_size * hidden_size

b1: hidden_size

W2: hidden_size * output_size

b2: output_size

Cache arrays prevent recomputation during backprop

Forward Pass – Step by Step
z1 = W1 · x + b1
a1 = sigmoid(z1)

z2 = W2 · a1 + b2
a2 = sigmoid(z2)   ← prediction

Flow diagram (ASCII):
   x (input)
      │
      ▼
   W1·x + b1  ──► z1 ──► sigmoid() ──► a1
                                          │
                                          ▼
                                       W2·a1 + b2 ──► z2 ──► sigmoid() ──► a2 (prediction)


Code snippet:

for (int i = 0; i < hidden_size; i++) {
    nn->z1[i] = nn->b1[i];
    for (int j = 0; j < input_size; j++)
        nn->z1[i] += nn->W1[j * hidden_size + i] * input[j];
    nn->a1[i] = sigmoid(nn->z1[i]);
}

 Backpropagation & Gradient Descent
 
 Loss function (MSE):
 L = (a2 - target)²

 Chain rule for output layer (W2):

 ∂L/∂W2 = ∂L/∂a2 * ∂a2/∂z2 * ∂z2/∂W2
       = (a2 - target) * sigmoid'(z2) * a1

Chain rule for hidden layer (W1):

∂L/∂W1 = ∂L/∂a2 * ∂a2/∂z2 * ∂z2/∂a1 * ∂a1/∂z1 * ∂z1/∂W1

Weight update (SGD):
W = W - learning_rate * ∂L/∂W

ASCII diagram – backward flow:
Error δ2  ←  (a2 - target) * sigmoid'(z2)
   │
   └──► δ1  ←  (W2ᵀ · δ2) * sigmoid'(z1)
          │
          └──► gradients for W1, b1

Training Loop (Pseudocode)
for epoch = 1 to epochs:
    total_loss = 0
    for each sample (x, target):
        forward(x)
        loss = (prediction - target)²
        total_loss += loss
        backward(x, target)   // compute gradients
        update_weights_and_biases()
    if epoch % 1000 == 0:
        print average_loss
