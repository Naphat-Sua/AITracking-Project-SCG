import os.path
import webbrowser

import numpy as np
import pandas as pd

from route_way import run_genetic_algorithm

if __name__ == "__main__":
    # all_route = run_genetic_algorithm(generations=5000, population_size=100)
    all_route = run_genetic_algorithm(generations=500, population_size=100)
    df = pd.DataFrame(data=all_route)
    df.to_csv("debug_all_route.csv")

    def list2String(list_data):
        return "['" + "','".join(map(str, list_data)) + "']"

    with open("Show.template", "r", encoding="utf-8") as f:
        html_template = f.read()

    route_str = "["
    for i in range(0, len(all_route) - 1):
        route_str += list2String(all_route[i]) + ","
    route_str += list2String(all_route[len(all_route) - 1]) + "]"

    start_str = list2String(df[0].values)
    last_col = np.shape(all_route)[1] - 1
    end_str = list2String(df[last_col].values)

    html_code = html_template % (start_str, end_str, route_str)

    file_name = "show_routing.html"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(html_code)

    # If your system sets IE as default, it may not show the html in your webbrowser
    webbrowser.open("file://" + os.path.realpath(file_name))
