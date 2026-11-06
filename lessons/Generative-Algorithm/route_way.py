# Genetic algorithm that computes the (approximately) optimal road trip.
# Borrowed codes from:
# http://www.randalolson.com/2015/03/10/computing-the-optimal-road-trip-across-europe/
# http://nbviewer.jupyter.org/github/rhiever/Data-Analysis-and-Machine-Learning-Projects/blob/master/optimal-road-trip/Computing%20the%20optimal%20road%20trip%20across%20the%20U.S..ipynb
# Thanks!

import random

import pandas as pd

WAYPOINT_CSV = "my-waypoints-dist-dur.csv"


def load_waypoints(csv_file=WAYPOINT_CSV):
    """Read the waypoint pair CSV (produced by Get-Data.py).

    Returns (waypoint_distances, waypoint_durations, all_waypoints) where the
    first two are dicts keyed by frozenset({waypoint1, waypoint2}).
    """
    waypoint_distances = {}
    waypoint_durations = {}
    all_waypoints = set()

    waypoint_data = pd.read_csv(csv_file, encoding="utf-8")
    for _, row in waypoint_data.iterrows():
        waypoint_distances[frozenset([row.waypoint1, row.waypoint2])] = row.distance_m
        waypoint_durations[frozenset([row.waypoint1, row.waypoint2])] = row.duration_s
        all_waypoints.update([row.waypoint1, row.waypoint2])

    return waypoint_distances, waypoint_durations, all_waypoints


def compute_fitness(solution, waypoint_distances):
    """Return the total distance traveled on the current road trip.

    The genetic algorithm will favor road trips that have shorter
    total distances traveled.
    """
    solution_fitness = 0.0
    # index begins at 1, so the trip does not loop back to the start
    for index in range(1, len(solution)):
        waypoint1 = solution[index - 1]
        waypoint2 = solution[index]
        solution_fitness += waypoint_distances[frozenset([waypoint1, waypoint2])]

    return solution_fitness


def generate_random_agent(all_waypoints):
    """Create a random road trip from the waypoints."""
    new_random_agent = list(all_waypoints)
    random.shuffle(new_random_agent)
    return tuple(new_random_agent)


def mutate_agent(agent_genome, max_mutations=3):
    """Apply 1 - `max_mutations` point mutations to the given road trip.

    A point mutation swaps the order of two waypoints in the road trip.
    """
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)

    for _ in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1

        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = (
            agent_genome[swap_index2],
            agent_genome[swap_index1],
        )

    return tuple(agent_genome)


def shuffle_mutation(agent_genome):
    """Apply a single shuffle mutation to the given road trip.

    A shuffle mutation takes a random sub-section of the road trip
    and moves it to another location in the road trip.
    """
    agent_genome = list(agent_genome)

    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)

    genome_subset = agent_genome[start_index:start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]

    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]

    return tuple(agent_genome)


def generate_random_population(pop_size, all_waypoints):
    """Generate a list with `pop_size` number of random road trips."""
    return [generate_random_agent(all_waypoints) for _ in range(pop_size)]


def run_genetic_algorithm(generations=5000, population_size=100,
                          csv_file=WAYPOINT_CSV, seed=None):
    """The core of the Genetic Algorithm.

    `generations` and `population_size` must be a multiple of 10.
    Pass `seed` to make a run reproducible.
    """
    if seed is not None:
        random.seed(seed)

    waypoint_distances, _, all_waypoints = load_waypoints(csv_file)

    all_route = []
    population_subset_size = int(population_size / 10.0)
    generations_10pct = int(generations / 10.0)

    # Create a random population of `population_size` number of solutions.
    population = generate_random_population(population_size, all_waypoints)

    # For `generations` number of repetitions...
    for generation in range(generations):
        # Compute the fitness of the entire current population
        population_fitness = {}

        for agent_genome in population:
            if agent_genome in population_fitness:
                continue
            population_fitness[agent_genome] = compute_fitness(agent_genome, waypoint_distances)

        # Take the top 10% shortest road trips and produce offspring each from them
        new_population = []
        ranked_population = sorted(population_fitness, key=population_fitness.get)
        for rank, agent_genome in enumerate(ranked_population[:population_subset_size]):
            if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                print(
                    "Generation %d best: %d | Unique genomes: %d"
                    % (generation, population_fitness[agent_genome], len(population_fitness))
                )
                all_route.append(agent_genome)

            # Create 1 exact copy of each of the top road trips
            new_population.append(agent_genome)

            # Create 2 offspring with 1-3 point mutations
            for _ in range(2):
                new_population.append(mutate_agent(agent_genome, 3))

            # Create 7 offspring with a single shuffle mutation
            for _ in range(7):
                new_population.append(shuffle_mutation(agent_genome))

        # Replace the old population with the new population of offspring
        population = new_population

    return all_route
