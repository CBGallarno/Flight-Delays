# Chris Gallarno and Vyann Chen
# Dataset: https://www.kaggle.com/usdot/flight-delays
# CS 484 Final Project
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV


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
              "month day day_of_week flight_time departure_time distance airline origin_airport dest_airport")
        exit(1)
    month_in, day_in, dow_in, time_in, arrv_in, dist_in, airline_in, dept_airport_in, arr_airport_in = sys.argv[1:]

    flights, airlines, airports = get_data(int(month_in), int(day_in))

    plot_data = flights[flights["DEPARTURE_DELAY"] < 300]
    plt.scatter(plot_data["SCHEDULED_DEPARTURE"], plot_data["DEPARTURE_DELAY"], s=0.2)
    plt.xlabel("Departure Time")
    plt.ylabel("Length Of Delay")
    plt.show()
    groups = plot_data.groupby("AIRLINE")
    plt.boxplot([data["DEPARTURE_DELAY"] for gr, data in groups], sym="", labels=groups.groups.keys())
    plt.xlabel("Airline")
    plt.ylabel("Length Of Delay")
    plt.show()
    print("plot")

    # one hot encoding
    flights = pd.get_dummies(flights)

    pre_flight_data = flights.drop(columns=["DEPARTURE_DELAY", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED", "MONTH", "DAY"])
    flight_result = flights[["DEPARTURE_DELAY", "ARRIVAL_DELAY", "DIVERTED", "CANCELLED"]]

    flight_input = pd.DataFrame(columns=pre_flight_data.columns, index=range(0, 24)).fillna(0)
    # flight_input["MONTH"] = float(month_in)
    # flight_input["DAY"] = float(day_in)
    flight_input["DAY_OF_WEEK"] = float(dow_in)
    flight_input["SCHEDULED_TIME"] = float(time_in)
    flight_input["DISTANCE"] = float(dist_in)
    flight_input["SCHEDULED_ARRIVAL"] = float(arrv_in)
    try:
        flight_input["AIRLINE_" + airline_in] = 1.0
        flight_input["ORIGIN_AIRPORT_" + dept_airport_in] = 1.0
        flight_input["DESTINATION_AIRPORT_" + arr_airport_in] = 1.0
    except KeyError:
        print()

    for i in range(0, 24):
        flight_input.at[i, "SCHEDULED_DEPARTURE"] = float(str(i) + "05")

    params = [{
        "alpha": [100],
        "l1_ratio": [0.1],
        "learning_rate": ["adaptive"],
        "loss": ["epsilon_insensitive"],
        "penalty": ["elasticnet"],
    }]
    search = GridSearchCV(SGDRegressor(), param_grid=params, scoring="r2", n_jobs=1, cv=5)

    search.fit(pre_flight_data, flight_result["DEPARTURE_DELAY"])
    print("Fit")
    prediction = search.predict(flight_input)

    print(pd.DataFrame({"predicted_delay": prediction, "departure_time": flight_input["SCHEDULED_DEPARTURE"]}))

    results = pd.DataFrame(search.cv_results_)
    results = results.sort_values(by=["rank_test_score"])
    results.to_csv("TuningResults.csv")
