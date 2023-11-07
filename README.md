Convolutional Neural Network (CNN) Based PyTorch model for the binary prediction of molecules as EGFR inhibitors using python script. 

*****

After installing the requirements in an Virtual Environment
or creating a new environment using conda by using "egfr_pred.yml" file
just open the command prompt or Terminal in the above created virtual environment
run the command given below
### Using Tensorflow to make prediction
python predict_CNN_EGFR.py test.smi


similarly, for any unknown molecule when just use the command by specifying the path
of "*.smi" file.
The supplied "test.smi" file is an Active molecule.

python predict_CNN_EGFR.py [specify_path]*.smi


The result will be displayed in the terminal as Molecule to be active or Inactive

For the trained model after Hyperparameter tuning, Achieved stats were as mentioned below:
Training Accuracy: 95.61%
Validation Accuracy: 86.21%





 
