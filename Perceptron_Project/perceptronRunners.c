#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

// USER ADJUSTED VARIABLES
#define DATA_SIZE 16          // Amount of Data Points
#define INPUT_SIZE 3          // Number of Different Inputs
#define TRAINING_SIZE 15      // How much of data to use for training
#define TESTING_SIZE 5        // How much of data to use for testing
#define EPOCHS 10000          // Amount of Times to Go Through Entire Dataset
#define LEARNING_RATE 0.0000025    // How Fast Weights change based on Error
#define PRINT_INTERVAL 5      // How Often to Print Results (in Epochs)
#define MIN_STOPPING_EPOCH 50 // Minimum Epochs before Early Stopping can Occur
bool printResults = true;     // Whether to Print Results at start of each Epoch
bool normalizeData = false;    // Whether to scale data between 0 and 1
bool earlyStopping = true;    // Whether to Stop Training if Error stops decreasing

// INPUTS: Height, Weight, Age
float inputs[DATA_SIZE][INPUT_SIZE] = {
            {186, 69, 25},  //Jakob Ingebrigston
            {187, 70, 28}, //Josh Kerr
            {172, 58, 30}, //Ronald Kwemoi
            {177, 62, 28}, //Grant Fisher
            {170, 57, 24}, //Jacob Krop
            {178, 61, 28}, //Mohamed Katir
            {185, 58, 28}, //Yomif Kejelcha
            {171, 56, 31}, //Hagos Gebrhiwet
            {170, 53, 26}, //Selemon Barega
            {171, 60, 32}, //Muktar Edris
            {183, 61, 29}, //Joshua Cheptegei
            {175, 57, 35}, //Paul Chemlimo
            {188, 70, 30}, //Stweart McSweyn
            {185, 63, 26}, //Yared Nuguse
            {187, 72, 27}, //Narve Gilje
            {175, 64, 34}}; //Andrew Butchart

// Ex. Result Price ($)
float label[] = {768.45, 803.78, 782.56, 766.96, 765.71, 765.01, 758.95, 756.73, 763.02, 774.83, 755.36, 777.55, 776.07, 820.62, 785.38, 786.211};

int main() {

    // Generate Random Weights for each input and bias
    float weights[INPUT_SIZE];
    for (int i = 0; i < INPUT_SIZE; i++) weights[i] = (float)rand() / RAND_MAX;

    float maxValues[INPUT_SIZE] = {0, 0, 0};
    float b = 0, eTotal = 0, result = 0, eAvg = 0, lastEAvg = 1000;
    int i = 0, epoch = 0;

    clock_t start = clock();

    if (normalizeData) {
        // Find max of each input
        for (int i = 0; i < DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                if (inputs[i][j] > maxValues[j])
                    maxValues[j] = inputs[i][j];
            }
        }
        // Divide all datasets by max
        for (int i = 0; i < DATA_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                inputs[i][j] /= maxValues[j];
            }
        }
    }

    if (printResults) printf("Goal Output %.2f\n", label[DATA_SIZE - 1]);

    for (epoch = 1; epoch <= EPOCHS; epoch++) {

        eAvg = 0;

        for (i = 0; i < DATA_SIZE; i++) {

            // Calculate Predicted Price
            result = 0;
            for (int j = 0; j < INPUT_SIZE; j++) result += inputs[i][j] * weights[j];
            result += b;

            // Calculate Error
            eTotal = label[i] - result;

            // Calculate Abs Average Error
            eAvg += fabs(eTotal / label[i]) * 100;

            // Change Weights
            for (int j = 0; j < INPUT_SIZE; j++) weights[j] += LEARNING_RATE * eTotal * inputs[i][j];

            // Change Bias
            b += LEARNING_RATE * eTotal;
        }

        eAvg /= DATA_SIZE;

        clock_t end = clock();

        double runtime = (double)(end - start) / CLOCKS_PER_SEC;

        // Calculate Result for Print Price
        result = 0;
        for (int j = 0; j < INPUT_SIZE; j++) result += inputs[DATA_SIZE - 1][j] * weights[j];
        result += b;

        if (epoch % PRINT_INTERVAL == 0 && printResults)
        {
            printf("Epoch: %i | Average Error: %.2f | Runtime: %.1f | Equation: %.0f = (%.2f * %.2f) + (%.2f * %.2f) + (%.2f * %.2f) + %.2f\n", epoch, eAvg, runtime * 1000, result, weights[0], inputs[DATA_SIZE - 1][0], weights[1], inputs[DATA_SIZE - 1][1], weights[2], inputs[DATA_SIZE - 1][2], b);
        }

        if (earlyStopping && lastEAvg - eAvg < 0.001 && epoch > MIN_STOPPING_EPOCH)
        {
            printf("Stopping Early - Error Improvement: %.4f\n", lastEAvg - eAvg);
            break;
        }

        lastEAvg = eAvg;
    }
    return 0;
}