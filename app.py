import json

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import requests
import os
import sklearn
from motapi.motdata import *
from datetime import datetime

app = Flask(__name__)

# Load the model from file
with open('/Users/matthewfard/PycharmProjects/flaskProject/code_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the model from file
with open('/Users/matthewfard/PycharmProjects/flaskProject/result_model2.pkl', 'rb') as f:
    result_model = pickle.load(f)

# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'tj5mFN2hPf7hRfcNq7WE9shwW09uuDG4NSgwmi73'
api_key2 = 'D65EuvBkZf5lQ7q6izGv19CgxPAq7qaI7oA98IMm'

url = 'https://driver-vehicle-licensing.api.gov.uk/vehicle-enquiry/v1/vehicles'
headers = {
    'x-api-key': api_key2,
    'Content-Type': 'application/json',
}

@app.route('/')
def hello_world():  # put application's code here
    print(type(model))
    return render_template('home.html')

@app.route('/results', methods=['POST'])
def render_results():
    reg = Registration(api_key)
    registration = request.form['reg'].replace(" ", "")
    reg_data = reg.get_data(registration)
    reg_data_ves = {
        'registrationNumber': registration,
    }
    response = requests.post(url, headers=headers, json=reg_data_ves)
    vehicle_data = response.json()
    engine_capacity = vehicle_data.get('engineCapacity', None)

    # Extract the details from the first vehicle in the list
    vehicle = reg_data[0]

    # Extract and save the required details as Python variables
    make = vehicle['make']
    vehicle_model = vehicle['model']
    fuel_type = vehicle['fuelType']
    engine = engine_capacity
    milage = request.form['milage']

    man_date = vehicle['firstUsedDate']
    man_date = pd.to_datetime(man_date, format='%Y.%m.%d')
    today_date = datetime.now()
    vehicle_age = (today_date - man_date).days / 365.25

    data = [[milage, make, vehicle_model, fuel_type, engine, vehicle_age]]
    df = pd.DataFrame(data, columns=['odometerValue', 'make', 'model', 'fuelType', 'engineSize', 'vehicle_age'])

    probabilities_result = result_model.predict_proba(df)
    sorted_indices_result = np.argsort(probabilities_result, axis=1)
    highest_prob_index_result = sorted_indices_result[:, -1]
    highest_prob_result = probabilities_result[np.arange(len(probabilities_result)), highest_prob_index_result]
    most_probable_class_result = result_model.classes_[highest_prob_index_result]

    probabilities = model.predict_proba(df)
    sorted_indices = np.argsort(probabilities, axis=1)
    highest_prob_index = sorted_indices[:, -1]
    second_highest_prob_index = sorted_indices[:, -2]
    third_highest_prob_index = sorted_indices[:, -3]
    fourth_highest_prob_index = sorted_indices[:, -4]
    fifth_highest_prob_index = sorted_indices[:, -5]
    #
    # # Get the actual probabilities
    highest_prob = probabilities[np.arange(len(probabilities)), highest_prob_index]
    second_highest_prob = probabilities[np.arange(len(probabilities)), second_highest_prob_index]
    third_highest_prob = probabilities[np.arange(len(probabilities)), third_highest_prob_index]
    fourth_highest_prob = probabilities[np.arange(len(probabilities)), fourth_highest_prob_index]
    fifth_highest_prob = probabilities[np.arange(len(probabilities)), fifth_highest_prob_index]

    # Get the class labels
    most_probable_class = model.classes_[highest_prob_index]
    second_most_probable_class = model.classes_[second_highest_prob_index]
    third_most_probable_class = model.classes_[third_highest_prob_index]
    fourth_most_probable_class = model.classes_[fourth_highest_prob_index]
    fifth_most_probable_class = model.classes_[fifth_highest_prob_index]

    fault_code_mapping = { '2.2 - Steering wheel or handlebar condition, Steering column or forks and yokes': 0, '8.4 - Fluid leaks including oil, engine coolant, and screen wash': 1,
                             '1.1 - Condition and operation of the brakes': 2,
                             '5.3 - Suspension system': 3,
                             '6.3 - Body, structure and attachments': 4,
                             '7.1 - Seat belts and supplementary restraint systems (SRS)': 5,
                             '2.3 - Steering play': 6,
                             '1.8 - Hydraulic brake fluid': 7,
                             '2.5 - Issues with suspension': 8,
                             '2.1 - Mechanical condition of steering': 9,
                             '2.4 - Suspension Springs, shock absorbers, or joints': 10,
                             '4.4 - Direction indicators and hazard warning lamps': 11,
                             '1.5 - Additional braking device (retarder) performance': 12,
                             '4.7 - Rear registration plate lamps': 13,
                             '1.7 - Electronic braking system (EBS)': 14,
                             '2.6 - Electronic power steering (EPS)': 15,
                             '1.4 - Parking brake performance and efficiency': 16,
                             '2.7 - Steering': 17,
                             '6.1 - Structure and attachments (including exhaust system and bumpers)': 18,
                             '5.1 - Axles, stub axles, and wheel bearings': 19,
                             '7.2 - Speed limiter (if required)': 20,
                             '1.2 - Service brake performance and efficiency': 21,
                             '1.3 - Secondary brake performance and efficiency': 22,
                             '7.7 - Audible warning (horn)': 23,
                             '6.2 - Body and interior (including doors and catches, seats and floor)': 24,
                             '5.4 - Axles, wheels, tyres and suspension': 25,
                             '7.3 - Anti-theft device': 26,
                             '1.9 - Brake pressure storage reservoirs': 27,
                             '1.0 - Condition of brakes': 28,
                             '7.4 - Vehicle safety equipment': 29,
                             '2.0 - Mechanical condition, steering wheel and column or handlebar': 30,
                             '6.6 - Body, structure and attachments': 31,
                             '7.8 - Speedometer': 32,
                             '7.12 - Electronic stability control (ESC)': 33,
                             '7.0 - Other equipment': 34,
                             '6.4 - Body, structure and attachments': 35,
                             '2.8 - Condition of the steering': 36,
                             '2.9 - Steering': 37,
                             '6.7 - Body, structure and attachments': 38}

    pass_fail_mapping = {'PASS': 0, 'FAIL': 1}

    # Creating a reverse mapping dictionary
    reverse_fault_code_mapping = {v: k for k, v in fault_code_mapping.items()}

    # Creating a reverse mapping dictionary
    reverse_pass_fail_mapping = {v: k for k, v in pass_fail_mapping.items()}

    def get_fault_code(class_label):
        return reverse_fault_code_mapping.get(class_label, "Unknown Fault Code")

    def get_pass_fail(class_label):
        return reverse_pass_fail_mapping.get(class_label, "Unknown Result")

    return render_template('results.html', carMake=make, carModel=vehicle_model, passOrFail=get_pass_fail(most_probable_class_result[0]), passOrFailprob=round(highest_prob_result[0]*100), first=get_fault_code(most_probable_class[0]), firstprob=round(highest_prob[0]*100), second=get_fault_code(second_most_probable_class[0]), secondprob=round(second_highest_prob[0]*100), third=get_fault_code(third_most_probable_class[0]), thirdprob=round(third_highest_prob[0]*100))


if __name__ == '__main__':
    app.run()
