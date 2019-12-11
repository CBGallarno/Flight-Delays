# Chris Gallarno and Vyann Chen
# Dataset: https://www.kaggle.com/usdot/flight-delays
# CS 484 Final Project

import pandas as pd


def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data


if __name__ == '__main__':
    airlines = read_csv("data/airlines.csv")
    airports = read_csv("data/airports.csv")
    flights = read_csv("data/flights.csv")

    flights = flights.drop(['YEAR', 'FLIGHT_NUMBER', 'TAIL_NUMBER', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], 1)
    flights.groupby("MONTH")
    print(flights)
