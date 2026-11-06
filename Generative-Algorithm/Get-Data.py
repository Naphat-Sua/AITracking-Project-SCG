# Build the distance/duration matrix between Thai provinces with the
# Google Maps Distance Matrix API.
# Borrowed idea from:
# http://www.randalolson.com/2015/03/10/computing-the-optimal-road-trip-across-europe/
#
# Requirements:
#   pip install tqdm googlemaps
#
# Enable the "Google Maps Distance Matrix API" on your Google account and
# export your key first (never commit an API key to source control):
#   export GOOGLE_MAPS_API_KEY="your-key-here"
# See https://github.com/googlemaps/google-maps-services-python#api-keys

import logging
import math
import os
import time
from itertools import combinations

import googlemaps
import pandas as pd
from tqdm import tqdm

LOG_FORMAT = "%(asctime)s: %(levelname)s: %(message)s"

HEAD_COLUMNS = ["waypoint1", "waypoint2", "distance_m", "duration_s"]
PROVINCE_CSV = "Province.csv"
ERROR_LOG = "error.txt"


def get_api_key():
    key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise SystemExit(
            "GOOGLE_MAPS_API_KEY environment variable is not set.\n"
            "Get a key from https://developers.google.com/maps/documentation/distance-matrix "
            "and run: export GOOGLE_MAPS_API_KEY='your-key-here'"
        )
    return key


def get_waypoints(num_province=10, file_tosave="my-waypoints-dist-dur.csv"):
    df_waypoints = pd.read_csv(PROVINCE_CSV, usecols=["province"], encoding="utf-8")
    df_sample = df_waypoints[0:num_province]
    all_waypoints = df_sample.values.ravel()
    assert len(all_waypoints) == num_province

    all_row = [[waypoint1, waypoint2] for (waypoint1, waypoint2) in combinations(all_waypoints, 2)]
    # total of (waypoint1, waypoint2) pairs = n!/(2*(n-2)!)
    assert len(all_row) == math.factorial(num_province) / (2 * math.factorial(num_province - 2))
    print("all waypoint:", len(all_row))

    gmaps = googlemaps.Client(key=get_api_key())

    all_data = []
    for i in tqdm(range(len(all_row)), ascii=True, desc="get distance and duration"):
        waypoint1, waypoint2 = all_row[i]
        try:
            route = gmaps.distance_matrix(
                origins=[waypoint1],
                destinations=[waypoint2],
                mode="driving",  # or "walking", "bicycling", ...
                language="Thai",  # or "English", ...
                units="metric",
            )
            # "distance" is in meters, "duration" is in seconds
            distance = route["rows"][0]["elements"][0]["distance"]["value"]
            duration = route["rows"][0]["elements"][0]["duration"]["value"]

            all_data.append([waypoint1, waypoint2, distance, duration])
            time.sleep(1)  # stay under the API rate limit
        except Exception:
            logging.exception("Error with finding the route between %s and %s.", waypoint1, waypoint2)
            with open(ERROR_LOG, "a", encoding="utf-8") as file:
                file.write(f"Error with finding the route between {waypoint1} and {waypoint2}.\n")

    df = pd.DataFrame(all_data, columns=HEAD_COLUMNS)
    df.to_csv(file_tosave, index=False, encoding="utf-8")


if __name__ == "__main__":
    # configure logging here, not at import: it mutates the root logger
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    # Limitations of the Google API: the maximum allowed waypoints is 23
    # plus the origin and destination.
    # routing possibilities for 20 provinces = 20! = 2432902008176640000
    # all waypoint pairs = 20!/(2 * (20-2)!) = 190
    get_waypoints(num_province=20)
    logging.info("=================Finish==================")
