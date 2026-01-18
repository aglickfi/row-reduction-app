# %%
import numpy as np
import streamlit as st
import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout
from IPython.display import display, clear_output, Math
from math import ceil

# %%
def format_coef(prevcoef, coef, var):

    #Properly formats system of equations depending on coefficient

    if var == 'x':
        if coef == 0:
            return "&"
        elif coef == 1:
            return f'&{var}'
        elif coef == -1:
            return f"-&{var}"
        else:
            return f"{coef}&{var}"

    if coef == 0:
        return ""
    elif coef == 1 and prevcoef == 0:
        return f"{var}"
    elif coef == 1:
        return f"+{var}"
    elif coef == -1:
        return f"-{var}"
    elif coef > 0 and prevcoef != 0:
        return f"+{coef}{var}"
    else:
        return f"{coef}{var}"

def matrix_to_tex(matrix):
    
    # Converts matrix to system of equations
    a, b, c = matrix[0]
    d, e, f = matrix[1]

    # Builds LaTeX string
    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"
    latex_str += f"{format_coef(1, a,'x')}{format_coef(a, b,'y')} = {c} \\\\"
    latex_str += f"{format_coef(1, d,'x')}{format_coef(d, e,'y')} = {f}"
    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def problem_generator(prob_num = 6, difficulty = "Small Numbers", low = -12, high = 12):

    problems = []

    for _ in range(prob_num):
        if difficulty == "Small Numbers":
            M = small_nice_2x3(low, high)
        else:
            M = big_nice_2x3(low, high)
        
        # Convert to LaTeX but return string instead of displaying
        latex = matrix_to_tex(M)
        problems.append(latex)

    return problems

def worksheet_generator(problems):

    header = r"""\documentclass[12pt]{article}
    \usepackage{amsmath}
    \begin{document}
    \section*{Row Reduction Practice Worksheet}
    """
    body = ""
    for i, p in enumerate(problems, start=1):
        body += f"Problem {i}: \\\\ {p} \\\\[1em]\n"
    
    footer = r"\end{document}"
    
    return header + body + footer


def big_nice_2x3(low, high):

    # Generates a (uniformly) random matrix

    while True:

        matrix = np.random.randint(low, high+1, size=(2,3))
        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        if det != 0:
            break



    # Ensures the final matrix has a nice RREF

    b1 = matrix[0][2]
    b2 = matrix[1][2]

    matrix[0][2] = det*b1
    matrix[1][2] = det*b2

    return matrix

def small_nice_2x3(low, high):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(2,3))

        # Checking for small determinant

        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    b1 = matrix[0][2]
    b2 = matrix[1][2]

    matrix[0][2] = det*b1
    matrix[1][2] = det*b2

    return matrix

#display(Math(matrix_to_tex(small_nice_2x3(-10,10))))


#def nice_matrix(m,n):



# %%
difficulty = st.selectbox(
    "Mode:",
    ["Small Numbers", "Large Numbers"]
)

generate = st.button("Generate system")

# %%
def show_problems_in_columns(problems):
    # Start LaTeX string
    latex_str = r"\LARGE\displaystyle \begin{array}{c c}"  # two centered columns
    
    for i in range(0, len(problems), 2):
        row = problems[i:i+2]
        # Each problem goes in a cell; leave blank if only one problem in row
        if len(row) == 1:
            row_str = f"\\text{{{i+1}.)}} {row[0]} & ~ "
        else:
            row_str = f"\\text{{{i+1}.)}} {row[0]} & \\text{{{i+2}.)}} {row[1]}"
        latex_str += row_str + r" \\[50pt]"  # vertical spacing between rows

    latex_str += r"\end{array}"
    
    st.latex(latex_str)

if generate:
    prob_num = 6
    problems = problem_generator(prob_num, difficulty=difficulty)
    show_problems_in_columns(problems)

# %%



