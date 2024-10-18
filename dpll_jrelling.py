#!/usr/bin/env python3

''' program to run the davis-putnam-logemann-loveland algorithm for a 2-sat solver '''

# imports
import csv
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# constants
FILENAME = 'data_2SATcnf_jrelling.csv'
OUTPUT = 'output_2SATcnf_jrelling.csv'

# functions
def dpll(num_var: int, num_clauses: int, problem: list[list]):
    ''' dpll implementation '''
    # debug printing
    # print(num_var)
    # print(num_clauses)
    # print(problem)

    start_time = time.time() 
    assignment = {x: None for x in range(1, num_var + 1)} # create initial assignment
    sat = 'S' if dpll_recursion(problem, assignment) else 'U'
    end_time = time.time()
    total_time = end_time - start_time 
    return sat, num_var, total_time

def dpll_recursion(problem: list[list], assignment: dict):
    ''' recursive structure for dpll '''
    # print(assignment) # debug
    # check if all clauses satisfied
    if all(is_satisfied(clause, assignment) for clause in problem):
        return True # SATISFIABLE

    # check if any clause unsatisfied --> means unsatisfied
    if any(is_unsatisfied(clause, assignment) for clause in problem):
        return False # UNSATISFIABLE

    # choose an unassaigned literal
    literal = None
    for var, value in assignment.items():
        if value is None:
            literal = var # what we want to assign
            break 
    # try assigning true and false to the literal
    for value in [True, False]:
        new_assignment = assignment.copy()
        new_assignment[literal] = value

        # simplify based on assignment and recurse
        simplified_problem = simplify(problem, literal, value)

        result = dpll_recursion(simplified_problem, new_assignment)
        if result:
            return True

    return False

def is_satisfied(clause: list, assignment: dict):
    ''' checks if a single clause is satisfied '''
    for literal in clause:
        if literal == 0: # end of line
            break
        var = abs(literal)
        value = assignment.get(var)
        if value is None:
            continue
        # check if literal in clause and its current assignment makes a clause true
        if (literal > 0 and value) or (literal < 0 and not value):
            return True

    return False


def is_unsatisfied(clause: list, assignment: dict):
    ''' checks if a single clause is definitely unsatisfied
        or not yet unsatisfied (not fully deduced) '''
    unsatisfied = True
    for literal in clause:
        if literal == 0: # end of line
            break
        var = abs(literal)
        value = assignment.get(var)
        if value is None:
            return False # not yet unsatisfied, unassigned variable
        # check if it is a satisfying assignment (like in is_satisfied), if so NOT unsatisfied
        if (literal > 0 and value) or (literal < 0 and not value):
            unsatisfied = False
            break
    return unsatisfied



def simplify(problem: list[list], literal: int | None, value: bool):
    ''' simplifies the problem based on an assignment '''
    simplified = [] # new problem structure

    for clause in problem:
        if (literal in clause) and value:
            continue # clause is satisfied, move to next (simplifying by removing the clause)
        if ((-1 * literal) in clause) and not value:
            continue # ^^^

        simplified_clause = []
        for lit in clause:
            if lit == 0: # end of line
                break
            # if clause contains negation of assigned literal, do not append lit back to simplify
            if lit != (-1 * literal):
                simplified_clause.append(lit)
        
        simplified_clause.append(0) # to signal end of line
        simplified.append(simplified_clause)
    
    return simplified

def generate_plot():
    data = pd.read_csv(OUTPUT)

    x = data['# variables']
    y = data['time (s)']
    sat = data['satisfiability']

    plt.figure(figsize=(10, 6))

    satisfiable = data[sat == 'S']
    plt.scatter(satisfiable['# variables'], satisfiable['time (s)'], color='green', label='Satisfiable')
    
    unsatisfiable = data[sat == 'U']
    plt.scatter(unsatisfiable['# variables'], unsatisfiable['time (s)'], color='red', label='Unsatisfiable')
    
    coefficients = np.polyfit(x, y, 1)  # Linear fit (1st degree polynomial)
    polynomial = np.poly1d(coefficients)
    best_fit_line = polynomial(x)

    plt.plot(x, best_fit_line, color='blue', linestyle='--')

    plt.xlabel('Number of Variables')
    plt.ylabel('Time (s)')
    # Dynamically set y-axis limits to better show variability
    y_min = min(y) - 0.05 * (max(y) - min(y))  # Padding for a little margin below
    y_max = max(y) + 0.05 * (max(y) - min(y))  # Padding for a little margin above
    plt.ylim(y_min, y_max)
    
    plt.title('Scatter Plot of Satisfiable vs Unsatisfiable')
    plt.legend()
    
    plt.savefig('./plot_jrelling.png', format='png')


def main():
    ''' get cnf problems from file '''
    problems = []
    curr_problem = []
    curr_clause = []

    with open(FILENAME, 'r') as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            if line[0] == 'c':
                if curr_problem:
                    problems.append(curr_problem)
                    curr_problem = []
            for x in line:
                if x.isdigit() or x.startswith('-'):
                    curr_clause.append(int(x))
                else:
                    curr_clause.append(x)
            curr_problem.append(curr_clause)
            curr_clause = []
    if curr_problem:
        problems.append(curr_problem)
    
    # in each problems[i], problems[i][1][2] is num_var, problems[i][1][3] is num_clauses
    
    # debug print
    # satisfiable, num_var, time = dpll(problems[0][1][2], problems[0][1][3], problems[0][2:])
    # print(satisfiable)
    # print(num_var)
    # print(time)

    ''' generate answers '''
    answers = [] # [[satisfiability, variables, time]]
    for problem in problems:
        # print (problem) # debug 
        answer = dpll(problem[1][2], problem[1][3], problem[2:])
        # print(answer) # debug
        answers.append(answer)

    # print(answers) # debug

    ''' create csv file '''
    with open(OUTPUT, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('satisfiability', '# variables', 'time (s)')) # header
        writer.writerows(answers)

    print(f'file {OUTPUT} created successfully')

    generate_plot()


if __name__ == '__main__':
    main()
