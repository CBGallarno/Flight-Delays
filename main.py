# Chris Gallarno and Vyann Chen
# Dataset: https://www.kaggle.com/usdot/flight-delays
# CS 484 Final Project
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

month_name = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data


def get_data(month_in, day_in):
    airlines = read_csv("data/airlines.csv")
    airports = read_csv("data/airports.csv")
    try:
        flights = read_csv("data/filtered_flights.csv")
    except FileNotFoundError:
        flights = read_csv("data/flights.csv")
        flights = flights.drop(['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY',
                                'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY',
                                'DEPARTURE_TIME',
                                'TAXI_OUT', 'WHEELS_OFF', 'ELAPSED_TIME', 'AIR_TIME', 'WHEELS_ON', 'TAXI_IN',
                                'ARRIVAL_TIME'
                                ], 1)
        flights = flights.dropna()
        flights.to_csv("data/filtered_flights.csv", index=False)

    print(flights["DEPARTURE_DELAY"].std())
    flights = flights[flights["MONTH"] == month_in]
    flights = flights[flights["DAY"] == day_in]
    return flights, airlines, airports


if __name__ == '__main__':
    if len(sys.argv) != 10:
        print("usage: python", sys.argv[0],
              "month day dow dept_time arrvival_time dist airline dept_airport arr_airport")
        exit(1)
    month_in, day_in, dow_in, dept_in, time_in, arrv_in, dist_in, dept_airport_in, arr_airport_in = sys.argv[1:]

    flights, airlines, airports = get_data(int(month_in), int(day_in))

    flights = pd.get_dummies(flights)

    pre_flight_data = flights.drop(columns=["DEPARTURE_DELAY", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED"])
    flight_result = flights[["DEPARTURE_DELAY", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED"]]

    flight_input = pd.DataFrame(columns=pre_flight_data.columns, index=range(len(airlines["IATA_CODE"]))).fillna(0)
    flight_input["MONTH"] = float(month_in)
    flight_input["DAY"] = float(day_in)
    flight_input["DAY_OF_WEEK"] = float(dow_in)
    flight_input["SCHEDULED_DEPARTURE"] = float(dept_in)
    flight_input["SCHEDULED_TIME"] = float(time_in)
    flight_input["DISTANCE"] = float(dist_in)
    flight_input["SCHEDULED_ARRIVAL"] = float(arrv_in)
    try:
        flight_input["ORIGIN_AIRPORT_" + dept_airport_in] = 1.0
        flight_input["DESTINATION_AIRPORT_" + arr_airport_in] = 1.0
    except KeyError:
        print()

    i = 0
    for airline in airlines["IATA_CODE"]:
        flight_input["AIRLINE_" + airline][i] = 1.0
        i += 1
    params = [{
        "alpha": [100, 1000],
        "l1_ratio": [0.1, 0.5, 0.9],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "loss": ["epsilon_insensitive", "huber"],
        "penalty": ["l1", "elasticnet"],
    }]
    search = GridSearchCV(SGDRegressor(), param_grid=params, scoring="r2", n_jobs=1)

    search.fit(pre_flight_data, flight_result["DEPARTURE_DELAY"])
    print("Fit")
    prediction = search.predict(flight_input)
    print(prediction)

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values(by=["rank_test_score"])
    results.to_csv("RES.csv")
