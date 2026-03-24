#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

//USER ADJUSTED VARIABLES
#define DATA_SIZE 20         //Amount of Data Points
#define INPUT_SIZE 3         //Number of Different Inputs
#define TRAINING_SIZE 15     //How much of data to use for training
#define TESTING_SIZE 5       //How much of data to use for testing
#define EPOCHS 10000         //Amount of Times to Go Through Entire Dataset
#define LEARNING_RATE 0.25   //How Fast Weights change based on Error
#define PRINT_INTERVAL 50     //How Often to Print Results (in Epochs)
#define MIN_STOPPING_EPOCH 50 //Minimum Epochs before Early Stopping can Occur
bool printResults = true;    //Whether to Print Results at start of each Epoch
bool normalizeData = true;   //Whether to scale data between 0 and 1
bool earlyStopping = true;   //Whether to Stop Training if Error stops decreasing

//INPUTS: Sq footage, bedrooms, yard size
float inputs[DATA_SIZE][INPUT_SIZE] =  {
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

//Ex. Result Price ($)
int label[] = {120000, 185000, 140000, 280000, 350000, 230000, 500000, 160000, 420000, 95000,
               270000, 470000, 200000, 330000, 75000, 255000, 390000, 155000, 540000, 300000};

int main() {

    //Generate Random Weights and Bias
    float weights[INPUT_SIZE];
    srand(time(NULL));
    for (int i = 0; i < INPUT_SIZE; i++) weights[i] = (float)rand() / RAND_MAX;
    float b = (float)rand() / RAND_MAX;

    float eTotal = 0, result = 0, eAvg = 0, lastEAvg = 1000;
    float maxValues[INPUT_SIZE] = {0, 0, 0};

    //START TIMING
    clock_t start = clock();

    //Normalize Data by Dividing Each Input by Max to Scale Between 0 and 1
    if (normalizeData) {
        //Find max of each input
        for (int i = 0; i < DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (inputs[i][j] > maxValues[j]) maxValues[j] = inputs[i][j];
            }
        }
        //Divide all datasets by max
        for (int i = 0; i < DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                inputs[i][j] /= maxValues[j];
            }
        }
    }

    if (printResults) printf("Goal Output: %i\n", label[DATA_SIZE - 1]);

    //Network Training Loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {

        //Reset Average Error for Epoch
        eAvg = 0;

        //Loop through each data point in training set
        for (int i = 0; i < DATA_SIZE; i++) {

            //Calculate Predicted Price
            result = 0;
            for (int j = 0; j < INPUT_SIZE; j++) result += inputs[i][j] * weights[j];
            result += b;

            //Activation Function: ReLU (Rectified Linear Unit)
            if (result < 0) result = 0;

            //Calculate Total Error
            eTotal = label[i] - result;

            //Calculate Abs Average Error
            eAvg += fabs(eTotal / label[i]);

            //Update Weights
            for (int j = 0; j < INPUT_SIZE; j++) {
                weights[j] += LEARNING_RATE * eTotal * inputs[i][j];
            }

            //Update Bias
            b += LEARNING_RATE * eTotal;

        }

        //Calculate Average Error for Epoch
        eAvg = (eAvg / DATA_SIZE) * 100;

        //End timing and calculate total runtime for epoch
        clock_t end = clock();
        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        //Print Results at set intervals
        if (epoch % PRINT_INTERVAL == 0 && printResults) {

            //Calculate Result with updated weights to print
            result = 0;
            for (int j = 0; j < INPUT_SIZE; j++) result += inputs[DATA_SIZE - 1][j] * weights[j];
            result += b;
            
            //Print Epoch, Average Error, Runtime, and Equation with Updated Weights
            printf("Epoch: %i | Average Error: %.2f | Runtime: %.1f | Equation: %.0f = ", epoch, eAvg, runtime * 1000, result);
            for (int z = 0; z < INPUT_SIZE; z++) {
                printf("(%.2f * %.2f) + ", weights[z], inputs[DATA_SIZE - 1][z]);
            }
            printf("%.2f\n", b);
        }

        //Stop early if error improvement is less than 0.001
        if (earlyStopping && lastEAvg - eAvg < 0.001 && epoch > MIN_STOPPING_EPOCH) {
            printf("Stopping Early - Error Improvement: %.4f\n", lastEAvg - eAvg);
            break;
        }

        lastEAvg = eAvg;
    }
    return 0;
}