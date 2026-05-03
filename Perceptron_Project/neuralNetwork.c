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
#define EPOCHS 500000         // Amount of Times to Go Through Entire Dataset
#define LEARNING_RATE 0.00000002    // How Fast Weights change based on Error
#define PRINT_INTERVAL 50000     // How Often to Print Results (in Epochs)
#define MIN_STOPPING_EPOCH 50 // Minimum Epochs before Early Stopping can Occur
#define LAYER_1_SIZE 15        // Number of Neurons in First Layer (Hidden Layer)
bool normalizeData = true;    // Whether to scale data between 0 and 1
bool earlyStopping = false;    // Whether to Stop Training if Error stops decreasing

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
int y[] = {120000, 185000, 140000, 280000, 350000, 230000, 500000, 160000, 420000, 95000, 270000, 470000, 200000, 330000, 75000, 255000, 390000, 155000, 540000, 300000};

//Non-Linear Labels
//int y[] = {95000, 210000, 125000, 480000, 890000, 370000, 2100000, 175000, 1400000, 72000, 460000, 1850000, 240000, 750000, 52000, 420000, 1150000, 162000, 2800000, 580000};

int main() {

    // Normalize Data by Dividing Each Input by Max to Scale Between 0 and 1
    if (normalizeData) {

        float maxValues[INPUT_SIZE] = {0, 0, 0};

        // Find max of each input
        for (int i = 0; i < DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (x[i][j] > maxValues[j]) maxValues[j] = x[i][j];
            }
        }

        // Divide all datasets by max
        for (int i = 0; i < DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                x[i][j] /= maxValues[j];
            }
        }
    }

    //Seed Random Number Generator
    srand(time(NULL));

    //Weights
    float W1[LAYER_1_SIZE][INPUT_SIZE];
    float W2[LAYER_1_SIZE];

    //Bias
    float B1[LAYER_1_SIZE];
    float B2 = (float) rand() / RAND_MAX;

    //Preactiviation Value
    float Z1[LAYER_1_SIZE];
    float Z2;

    //Activation Value
    float A1[LAYER_1_SIZE];

    float eTotal = 0, eTrainingAvg = 0, lastEAvg = 1000, eTestingAvg = 0, hiddenError = 0;

    //Initialize Weights and Bias with small random values between -0.005 and 0.005
    for (int i = 0; i < LAYER_1_SIZE; i++) {
        B1[i] = ((float)rand() / RAND_MAX) * 0.01 - 0.005;
        W2[i] = ((float)rand() / RAND_MAX) * 0.01 - 0.005;
        for (int j = 0; j < INPUT_SIZE; j++) {
            W1[i][j] = ((float)rand() / RAND_MAX) * 0.01 - 0.005;
        }
    }

    // START TIMING
    clock_t start = clock();

    //Network Training Loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {

        //Reset Average Error for Epoch
        eTrainingAvg = 0;

        //Loop through each data point in training set
        for (int i = 0; i < TRAINING_SIZE; i++) {

            //Calculate Result for each neuron in first layer
            for (int j = 0; j < LAYER_1_SIZE; j++) {

                //Calculate Predicted Price
                Z1[j] = 0;
                for (int k = 0; k < INPUT_SIZE; k++) Z1[j] += x[i][k] * W1[j][k];
                Z1[j] += B1[j];

                //Activation Function: ReLU (Rectified Linear Unit)
                A1[j] = Z1[j];
                if (A1[j] < 0) A1[j] = 0;
            }

            //Calculate Result for Output Neuron
            Z2 = B2;
            for (int j = 0; j < LAYER_1_SIZE; j++) Z2 += A1[j] * W2[j];

            //Calculate Total Error
            eTotal = y[i] - Z2;

            //Calculate Abs Average Error
            eTrainingAvg += fabs(eTotal / y[i]);

            //Update Layer 1 Weights and Bias
            for (int j = 0; j < LAYER_1_SIZE; j++) {

                hiddenError = eTotal * W2[j] * (A1[j] > 0 ? 1.0f : 0.0f);

                B1[j] += LEARNING_RATE * hiddenError;

                for (int k = 0; k < INPUT_SIZE; k++) {
                    W1[j][k] += LEARNING_RATE * hiddenError * x[i][k];
                }
            }

            //Update Layer 2 Weights
            for (int j = 0; j < LAYER_1_SIZE; j++) {
                W2[j] += LEARNING_RATE * eTotal * A1[j];
            }
            //Update Layer 2 Bias
            B2 += LEARNING_RATE * eTotal;
        }

        //Calculate Average Error for Epoch
        eTrainingAvg = (eTrainingAvg / TRAINING_SIZE) * 100;

        //End timing and calculate total runtime for epoch
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        //Print Results at set intervals
        if (epoch % PRINT_INTERVAL == 0) {

            //Print Epoch, Average Error, and Runtime
            printf("Epoch: %i | Average Error: %.2f | Runtime: %.1f | Result: %.1f\n", epoch, eTrainingAvg, runtime * 1000, Z2);
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

        //Calculate Result for each neuron in first layer
        for (int j = 0; j < LAYER_1_SIZE; j++) {

            // Calculate Predicted Price
            Z1[j] = 0;
            for (int k = 0; k < INPUT_SIZE; k++)
                Z1[j] += x[i][k] * W1[j][k];
            Z1[j] += B1[j];

            // Activation Function: ReLU (Rectified Linear Unit)
            A1[j] = Z1[j];
            if (A1[j] < 0) A1[j] = 0;
        }

        // Calculate Result for Output Neuron
        Z2 = B2;
        for (int j = 0; j < LAYER_1_SIZE; j++) Z2 += A1[j] * W2[j];

        // Calculate Abs Average Error
        eTestingAvg += fabs((y[i] - Z2) / y[i]);
    }

    // Calculate Average Error for Epoch
    eTestingAvg = (eTestingAvg / TESTING_SIZE) * 100;

    printf("Training Final Error: %.2f | Testing Average Error: %.2f", eTrainingAvg, eTestingAvg);

    return 0;
}