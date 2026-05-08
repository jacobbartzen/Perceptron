#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

//USER ADJUSTED VARIABLES
#define DATA_SIZE 20          // Amount of Data Points
#define INPUT_SIZE 3          // Number of Different Inputs
#define TRAINING_SIZE 15      // How much of data to use for training
int TESTING_SIZE = DATA_SIZE - TRAINING_SIZE;
#define EPOCHS 50000         // Amount of Times to Go Through Entire Dataset
#define LEARNING_RATE 0.05    // How Fast Weights change based on Error
#define PRINT_INTERVAL 10000     // How Often to Print Results (in Epochs)
#define MIN_STOPPING_EPOCH 50 // Minimum Epochs before Early Stopping can Occur
bool normalizeData = true;        //Whether to scale data between 0 and 1
bool earlyStopping = false;       //Whether to Stop Training if Error stops decreasing
int neuronLayers[] = {50, 25, 15, 5, 1};  //Array of Neuron Counts for Each Layer
#define layers 5                  //Number of Layers in Network (including output layer)
#define maxNeurons 50             //Max Number of Neurons in a Layer (for array sizing)

//INPUTS: Sq footage, bedrooms, yard size
float x[DATA_SIZE][INPUT_SIZE] = {
    {850, 1, 500},
    {1200, 2, 1000},
    {950, 2, 750},
    {1800, 3, 1500},
    {2200, 4, 2000},
    {1500, 3, 1200},
    {3000, 5, 3000},
    {1100, 2, 800},
    {2600, 4, 2500},
    {700, 1, 400},
    {1750, 3, 1300},
    {2900, 4, 2800},
    {1350, 2, 900},
    {2100, 3, 1600},
    {500, 1, 600},
    {1650, 3, 1400},
    {2400, 4, 2200},
    {1050, 2, 850},
    {3200, 5, 3500},
    {1900, 3, 1800}};

// Ex. Result Price ($)
//Linear Labels
float y[] = {120000, 185000, 140000, 280000, 350000, 230000, 500000, 160000, 420000, 95000, 270000, 470000, 200000, 330000, 75000, 255000, 390000, 155000, 540000, 300000};

//Non-Linear Labels
//float y[] = {95000, 210000, 125000, 480000, 890000, 370000, 2100000, 175000, 1400000, 72000, 460000, 1850000, 240000, 750000, 52000, 420000, 1150000, 162000, 2800000, 580000};

int main() {

    // Normalize Data by Dividing Each Input by Max to Scale Between 0 and 1
    if (normalizeData) {

        float maxValues[INPUT_SIZE + 1] = {0, 0, 0, 0};

        // Find all maxes
        for (int i = 0; i < DATA_SIZE; i++) {

            //Find max of inputs
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (x[i][j] > maxValues[j]) maxValues[j] = x[i][j];
            }

            //Find max of outputs
            if (y[i] > maxValues[INPUT_SIZE]) maxValues[INPUT_SIZE] = y[i];
        }

        //Divide data by max
        for (int i = 0; i < DATA_SIZE; i++) {

            //Divide inputs by max
            for (int j = 0; j < INPUT_SIZE; j++) {
                x[i][j] /= maxValues[j];
            }

            //Divide output by max
            y[i] /= maxValues[INPUT_SIZE];
        }
    }

    //Seed Random Number Generator
    srand(time(NULL));

    //Weights
    float W[layers][maxNeurons][maxNeurons];

    //Bias
    float B[layers][maxNeurons];

    //Preactiviation Value
    float Z[layers][maxNeurons];

    //Activation Value
    float A[layers][maxNeurons];

    //Delta for Backpropagation
    float D[layers][maxNeurons];

    float eTotal = 0, eTrainingAvg = 0, lastEAvg = 1000, eTestingAvg = 0, hiddenError = 0, scale = 0;

    //Initialize Weights and Bias with small random values between -0.005 and 0.005
    for (int i = 0; i < layers; i++) {
        int prevSize = (i == 0) ? INPUT_SIZE : neuronLayers[i - 1];
        scale = sqrt(2.0f / prevSize);
        for (int j = 0; j < neuronLayers[i]; j++) {
            B[i][j] = 0; // bias initialized to 0 is standard with He init
            for (int k = 0; k < prevSize; k++) {
                W[i][j][k] = (((float)rand() / RAND_MAX) * 2 - 1) * scale;
            }
        }
    }

    //START TIMING
    clock_t start = clock();

    //Network Training Loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {

        //Reset Average Error for Epoch
        eTrainingAvg = 0;

        //Loop through each data point in training set
        for (int i = 0; i < TRAINING_SIZE; i++) {

            //Forward Pass
            //For each layer
            for (int j = 0; j < layers; j++) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    Z[j][k] = B[j][k];

                    //If first layer, use inputs
                    if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) Z[j][k] += x[i][z] * W[j][k][z];

                    //Else, use outputs from previous layer
                    else for (int z = 0; z < neuronLayers[j - 1]; z++) Z[j][k] += A[j - 1][z] * W[j][k][z];

                    //Activation Function: Leaky ReLU (Rectified Linear Unit)
                    A[j][k] = (Z[j][k] > 0) ? Z[j][k] : 0.01f * Z[j][k];
                }
            }

            //Calculate Total Error
            eTotal = y[i] - Z[layers - 1][0];

            //Calculate Abs Average Error
            eTrainingAvg += fabs(eTotal / y[i]);

            //Backward Pass (Backpropagation)
            //For output layer, delta is total error
            D[layers - 1][0] = eTotal;

            //For each layer
            for (int j = layers - 2; j >= 0; j--) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    //Set Delta to 0
                    D[j][k] = 0.0f;

                    //Sum of Deltas from layer ahead * corresponding weights
                    for (int z = 0; z < neuronLayers[j + 1]; z++) {
                        D[j][k] += D[j + 1][z] * W[j + 1][z][k];
                    }

                    //Leaky ReLU Derivative
                    D[j][k] *= (Z[j][k] > 0) ? 1.0f : 0.01f;
                }
            }

            //Update Weights and Biases using Deltas
            //For each layer
            for (int j = 0; j < layers; j++) {

                //For each neuron
                for (int k = 0; k < neuronLayers[j]; k++) {

                    //Update Bias
                    B[j][k] += LEARNING_RATE * D[j][k];

                    //If first layer use inputs
                    if (j == 0) {
                        for (int z = 0; z < INPUT_SIZE; z++) {
                            W[j][k][z] += LEARNING_RATE * D[j][k] * x[i][z];
                        }
                    }

                    //Else use other neurons
                    else {
                        for (int z = 0; z < neuronLayers[j - 1]; z++) {
                            W[j][k][z] += LEARNING_RATE * D[j][k] * A[j - 1][z];
                        }
                    }
                }
            }
        }

        //Calculate Average Error for Epoch
        eTrainingAvg = (eTrainingAvg / TRAINING_SIZE) * 100;

        //End timing and calculate total runtime for epoch
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        //Print Results at set intervals
        if (epoch % PRINT_INTERVAL == 0) {

            //Print Epoch, Average Error, and Runtime
            printf("Epoch: %i | Average Error: %.2f | Runtime: %.1f | Result: %.3f\n", epoch, eTrainingAvg, runtime * 1000, Z[layers - 1][0]);

            //for (int j = 0; j < layers; j++) {
            //    for (int k = 0; k < neuronLayers[j]; k++) {
            //        printf("D[%d][%d] = %.6f | W[%d][%d][0] = %.6f\n", j, k, D[j][k], j, k, W[j][k][0]);
            //    }
            //}
        }

        //Stop early if error improvement is less than 0.001
        if (earlyStopping && lastEAvg - eTrainingAvg < 0.001 && epoch > MIN_STOPPING_EPOCH) {
            printf("Stopping Early - Error Improvement: %.4f\n", lastEAvg - eTrainingAvg);
            break;
        }

        lastEAvg = eTrainingAvg;
    }

    //Loop through testing data
    for (int i = TRAINING_SIZE; i < DATA_SIZE; i++) {

        // For each layer
        for (int j = 0; j < layers; j++) {

            // For each neuron
            for (int k = 0; k < neuronLayers[j]; k++) {

                Z[j][k] = B[j][k];

                // If first layer, use inputs
                if (j == 0) for (int z = 0; z < INPUT_SIZE; z++) Z[j][k] += x[i][z] * W[j][k][z];

                // Else, use outputs from previous layer
                else for (int z = 0; z < neuronLayers[j - 1]; z++) Z[j][k] += A[j - 1][z] * W[j][k][z];

                // Activation Function: ReLU (Rectified Linear Unit)
                A[j][k] = (Z[j][k] > 0) ? Z[j][k] : 0;
            }
        }

        // Calculate Abs Average Error
        eTestingAvg += fabs((y[i] - Z[layers - 1][0]) / y[i]);
    }

    // Calculate Average Error for Epoch
    eTestingAvg = (eTestingAvg / TESTING_SIZE) * 100;

    printf("Training Final Error: %.2f | Testing Average Error: %.2f", eTrainingAvg, eTestingAvg);

    return 0;
}