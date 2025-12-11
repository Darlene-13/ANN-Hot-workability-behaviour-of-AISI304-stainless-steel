
# USING ARTIFICAL NEURAL NETWORKS TO OBTAIN THE OPTIMAL INPUT PARAMETERS FOR PREDICTING THE FLOW STRESS OF AISI 304 STAINLESS STEEL

> This repository contains the implementation of an Artificial Neural Network (Multi layer Perceptron with Backpropation to predict the flow stress of AISI 304 Stainless Steel during hot deformation processes based on temperature, strain rate, and strain.)

> The project is based on the research paper: "Hot workability behaviour of AISI304 stainless steel: constitutive and
ANN modelling" by (Japheth Obiko  Brendon Mxolisi & Malatji Nicholus )
> 
> The link to the paper: [Hot workability behaviour of AISI304 stainless steel: constitutive and
ANN modelling](https://doi.org/10.1007/s12008-025-02436-x)

> This section of the project seeks to find the optimal input parameters that is Temperature and Strain Rate.
> 
> The test conditions were a deformation
temperature range of 950–1050℃ and a strain rate range of 0.1–15 s⁻¹. The study analysed the metal flow pattern and
compared the prediction accuracy of the Arrhenius, strain-compensation, and physical and Artificial Neural Networks
(ANN) models using statistical parameters: the correlation coefficient R and average absolute relative error AARE.
> 
> The results show that flow stress increases with a decrease in the deformation temperature and an increase in strain rate, and
vice versa. The predicted data obtained using the ANN model accurately tracks the experimental data throughout the
entire loading condition range. However, the constitutive model analyses show a marked deviation from experimental
data.
> 

**Model Used** : Multilayer Perceptron (MLP) with Backpropagation

## AISI 304 STAINLESS STEEL - FLOW STRESS PREDICTION

> - Project: Artificial Neural Network for Predicting Hot Deformation Behavior 
> 
> - Method: Multilayer Perceptron (MLP) with Backpropagation
> 
> - Activation Functions: tanh (hidden) + linear (output)
> 
> - Optimizer: Adam with Learning Rate = 0.01
> 
> - Loss Function: Mean Squared Error (MSE)

### The Core Problem 
The Project is working with **AISI 304 Stainless Steel** - a metal used in manufacturing (automotive, aerospace, industrial equipment).

**The Real Question:** When we heat this metal and squeeze it (deformation), how much force do we need?

**Why it matters:**
- Engineers need to know: "If I forge at 1000°C with a squeeze speed of 1 s⁻¹, how hard will the metal resist?"
- Getting this wrong = broken metal or wasted time/resources
- The paper shows you CAN predict this with high accuracy

### The Three Factors That Matter

1. **Temperature (T)** 
   - Cold metal = hard to squeeze (high resistance)
   - Hot metal = easy to squeeze (low resistance)
   - Range in study: 950-1050°C

2. **Strain Rate (ε̇)** - How fast you squeeze
   - Slow squeeze = metal can "relax" → lower resistance
   - Fast squeeze = metal can't relax → higher resistance
   - Range in study: 0.1 to 15 times per second

3. **Strain (ε)** - How much you've already squeezed
   - Initial squeezing = gets harder (work hardening)
   - After some squeezing = reaches stable state
   - Range in study: 0% to 70% deformation

### What We're Predicting
1. **Flow Stress (σ)** - The force per unit area needed to keep deforming the metal
- Higher flow stress = harder to deform
- Lower flow stress = easier to deform

2. **The Optimal Input Parameters**
- That is at what Temperature and Strain Rate do we get the best predictions for Flow Stress?
- We want to find the sweet spot where our ANN model predicts Flow Stress most accurately.

### Dependencies
1. Python 3 +
2. Core Libraries:
   - NumPy
   - Pandas
   - Scikit-learn
   - TensorFlow/Keras
   - Matplotlib/Seaborn
3. Jupyter Notebook (for interactive coding and visualization)
4. virtual environment (optional but recommended)

### SETTING UP THE PROJECT
1. Clone the repository:
   ```bash
   git clone
   Project URL: git@github.com:Darlene-13/ANN-Hot-workability-behaviour-of-AISI304-stainless-steel.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ANN-Hot-workability-behaviour-of-AISI304-stainless-steel
   ```
3. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### FILE STRUCTURE



### MACHINE LEARNING MODEL DEVELOPMENT
1. Data Interpolation and Extrapolation
- Our initial dataset had 60 data points from experiments, while this data was sufficient to train the model, I believe that increasing the number of data points
would improve the model's predictive accuracy and generalization capabilities. To achieve this, I employed interpolation and extrapolation techniques to generate additional data points within and beyond the original dataset's range.
- I used SciPy's `interp1d` function for interpolation and extrapolation. This function allows us to create a continuous function that fits the existing data points and can be used to estimate values at new points.
- By applying interpolation, I generated additional data points within the range of the original dataset. Extrapolation was used to estimate values beyond the existing data range, providing a more comprehensive dataset for training the ANN model.
  - This approach not only increased the dataset size but also helped capture the underlying trends and patterns in the data, which is crucial for improving the model's performance.

2. Data Analysis:
- Loaded and Explored the dataset, (our new dataset with interpolated and extrapolated data points)
- Performed Basic EDA (Exploratory Data Analysis) to understand the relationships between Temperature, Strain Rate, Strain, and Flow Stress.
- Visualized the data using scatter plots and heatmaps to identify trends and correlations.

3. Data Preprocessing:
- Normalized the input features (Temperature, Strain Rate, Strain) to ensure they are on a similar scale, which helps improve the training process of the ANN model.
- Split the dataset into training and testing sets to evaluate the model's performance.
- Implemented data augmentation techniques to enhance the diversity of the training data.


### DATA COLUMNS TO BE USED IN ANN
1. Inverse of Temperature (1/T): 
- This is required because flow stress follows Arrhenius equation: exp(-Q/RT)
- The inverse of temperature linearized this relationship
2. ln(Strain_Rate)
- Strain rate varies from 0.1 to 0.15 that is a huge range
- Finding its log makes it range from -2.3 to 2.7 this makes the range manageable
3. Strain

### Written by:
Darlene Wendy Nasimiyu