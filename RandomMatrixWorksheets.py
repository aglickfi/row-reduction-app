# %%
import numpy as np
import streamlit as st
import sympy as sp
from sympy import matrix2numpy
from math import ceil

# %%

# This group of functions generates the matrices to be row-reduced

def beginner_2x3(low = -5, high = 5):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(2,3))

        # Checking for small determinant

        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,2] *= det

    return matrix

def beginner_3x4(low = -5, high = 5):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(3,4))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 3, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,3] *= det

    return matrix

def beginner_4x5(low = -5, high = 5):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(4,5))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 4, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,4] *= det

    return matrix


def advanced_2x3(low = -20, high = 20):

    # Generates a (uniformly) random matrix

    while True:

        matrix = np.random.randint(low, high+1, size=(2,3))
        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        if det != 0:
            break

    # Ensures the final matrix has a nice RREF

    matrix[:,2] *= det

    return matrix

def advanced_3x4(low = -20, high = 20):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(3,4))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 3, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,3] *= det

    return matrix

def advanced_4x5(low = -20, high = 20):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(4,5))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 4, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,4] *= det

    return matrix

# These functions create singular matrices

def singularize_inf(matrix):

    n_eq = matrix.shape[0]
    n_cols = matrix.shape[1]

    # Determines the number of free variables

    n_free = np.random.randint(low = 1, high = n_eq)

    # Chooses random coefficients for linear combination

    while True:
        
        lc_coefs = np.random.randint(size = (n_free, n_eq-n_free), high = 6, low = -5)

        if np.all(np.any(lc_coefs != 0, axis = 1)):

            break

    for i in range(1,n_free+1):
        matrix[-i,:] = np.zeros(n_cols)
        for j in range(n_eq - n_free):
        
            matrix[-i,:] += lc_coefs[i-1,j]*matrix[j,:]

    return matrix

def singularize_no(matrix):

    n_eq = matrix.shape[0]
    n_cols = matrix.shape[1]

    # Chooses random coefficients for linear combination

    while True:
        
        lc_coefs = np.random.randint(size = n_eq-1, high = 6, low = -5)

        if np.any(lc_coefs != 0):

            break

    matrix[-1,:] = np.zeros(n_cols)

    for i in range(n_eq - 1):
        
        matrix[-1,:] += lc_coefs[i]*matrix[i,:]

    # Ensures no solution (most of the time)

    matrix[-1][-1] += np.random.randint(high = 10, low = -10)

    return matrix

# These functions generate systems with no solutions or infinitely many solutions

def beginner_2x3_sing(low = -5, high = 5):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(2,3))

        # Checking for small determinant

        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,2] *= det

    # Chooses if the matrix will have infinitely many solutions or no solution
    
    p = np.random.rand()

    if p < .33:

        singularize_inf(matrix)

    else:

        singularize_no(matrix)

    return matrix

def beginner_3x4_sing(low = -5, high = 5):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(3,4))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 3, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,3] *= det

    # Chooses if the matrix will have infinitely many solutions or no solution
    
    p = np.random.rand()

    if p < .33:

        singularize_inf(matrix)

    else:

        singularize_no(matrix)

    return matrix

def beginner_4x5_sing(low = -5, high = 5):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(4,5))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 4, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if np.abs(det) <= 5 and det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,4] *= det

    # Chooses if the matrix will have infinitely many solutions or no solution
    
    p = np.random.rand()

    if p < .33:

        singularize_inf(matrix)

    else:

        singularize_no(matrix)

    return matrix

def advanced_2x3_sing(low = -20, high = 20):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(2,3))

        # Checking for small determinant

        det = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]

        if det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,2] *= det

    # Chooses if the matrix will have infinitely many solutions or no solution
    
    p = np.random.rand()

    if p < .33:

        singularize_inf(matrix)

    else:

        singularize_no(matrix)

    return matrix

def advanced_3x4_sing(low = -20, high = 20):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(3,4))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 3, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,3] *= det

    # Chooses if the matrix will have infinitely many solutions or no solution
    
    p = np.random.rand()

    if p < .33:

        singularize_inf(matrix)

    else:

        singularize_no(matrix)

    return matrix

def advanced_4x5_sing(low = -20, high = 20):

    # Generates a (uniformly) random matrix

    while True: 

        matrix = np.random.randint(low, high+1, size=(4,5))

        # Checking for small determinant

        coef_matrix = np.delete(matrix, 4, axis = 1)

        det = int(round(np.linalg.det(coef_matrix)))

        if det != 0:

            break

    # Ensures the final matrix has a nice RREF
    
    matrix[:,4] *= det

    # Chooses if the matrix will have infinitely many solutions or no solution
    
    p = np.random.rand()

    if p < .33:

        singularize_inf(matrix)

    else:

        singularize_no(matrix)

    return matrix

# These functions help manage the conventional notation used in systems of equations

def format_coef2x3(prevcoef, coef, var):

    # Properly formats system of equations depending on coefficient

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
    
def format_coef3x4(prevcoef1,prevcoef2, coef, var):

    # Properly formats system of equations depending on coefficient

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
    elif coef == 1 and prevcoef1 == 0 and prevcoef2 == 0:
        return f"{var}"
    elif coef == 1:
        return f"+{var}"
    elif coef == -1:
        return f"-{var}"
    elif coef > 0 and (prevcoef1 != 0 or prevcoef2 != 0):
        return f"+{coef}{var}"
    else:
        return f"{coef}{var}"
    
def format_coef4x5(prevcoef1, prevcoef2, prevcoef3, coef, var):
    
    # Properly formats system of equations depending on coefficient

    if var == 'x':
        if coef == 0:
            return "&"
        elif coef == 1:
            return f"&{var}"
        elif coef == -1:
            return f"-&{var}"
        else:
            return f"{coef}&{var}"

    if coef == 0:
        return ""
    elif coef == 1 and prevcoef1 == 0 and prevcoef2 == 0 and prevcoef3 == 0:
        return f"{var}"
    elif coef == 1:
        return f"+{var}"
    elif coef == -1:
        return f"-{var}"
    elif coef > 0 and (prevcoef1 != 0 or prevcoef2 != 0 or prevcoef3 != 0):
        return f"+{coef}{var}"
    else:
        return f"{coef}{var}"

def arb_format_coef(prevcoef, coef, var):

    #Properly formats system of equations depending on coefficient

    if var == 'x_1':
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
    elif coef == 1 and prevcoef == 0 and var == 'x_2':
        return f"{var}"
    elif coef == 1:
        return f"+{var}"
    elif coef == -1:
        return f"-{var}"
    elif coef > 0:
        return f"+{coef}{var}"
    else:
        return f"{coef}{var}"
    
# These functions aid in solving the systems of equations and displaying their solutions

def system_solver(matrix):

    sp_matrix = sp.Matrix(matrix)

    sp_rref_matrix = sp_matrix.rref()[0]

    rref_matrix = sp.matrix2numpy(sp_rref_matrix)

    return rref_matrix

def singular_check(rref_matrix):

    naug_rref_matrix = rref_matrix[:,:-1]

    if np.any(np.all(naug_rref_matrix == 0, axis = 1)):

        for row in rref_matrix:

            if np.all(row[:-1] == 0) and row[-1] != 0:

                return 0
        
        # Uses rank-nullity to check for infinitely many solutions

        rank = np.sum(np.any(naug_rref_matrix != 0, axis = 1))
        n_vars = naug_rref_matrix.shape[1]

        free_vars = n_vars - rank
        
        if free_vars > 0:

            return 1
        
    return 2
    
# These functions convert numpy matrices to systems of equations in LaTeX

def matrix_to_tex2x3(matrix):
    
    # Converts matrix to system of equations
    a, b, c = matrix[0]
    d, e, f = matrix[1]

    # Builds LaTeX string
    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"
    latex_str += f"{format_coef2x3(1, a,'x')}{format_coef2x3(a, b,'y')} = {c} \\\\"
    latex_str += f"{format_coef2x3(1, d,'x')}{format_coef2x3(d, e,'y')} = {f}"
    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def solution_to_tex2x3(rref_matrix):

    check = singular_check(rref_matrix)

    if check == 0:

        return r"\text{\ No Solution}"
    
    elif check == 1:

        return r"\text{\ Infinitely Many Solutions}"


    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"
    latex_str += f"x = {rref_matrix[0][-1]} \\\\"
    latex_str += f"y = {rref_matrix[1][-1]}"
    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def matrix_to_tex3x4(matrix):

    a, b, c, d = matrix[0]
    e, f, g, h = matrix[1]
    i, j, k, l = matrix[2]

    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"

    # Row 1
    latex_str += (
        f"{format_coef3x4(0, 0, a, 'x')}"
        f"{format_coef3x4(a, 0, b, 'y')}"
        f"{format_coef3x4(a, b, c, 'z')}"
        f"= {d} \\\\"
    )

    # Row 2
    latex_str += (
        f"{format_coef3x4(0, 0, e, 'x')}"
        f"{format_coef3x4(e, 0, f, 'y')}"
        f"{format_coef3x4(e, f, g, 'z')}"
        f"= {h} \\\\"
    )

    # Row 3
    latex_str += (
        f"{format_coef3x4(0, 0, i, 'x')}"
        f"{format_coef3x4(i, 0, j, 'y')}"
        f"{format_coef3x4(i, j, k, 'z')}"
        f"= {l}"
    )

    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def solution_to_tex3x4(rref_matrix):

    check = singular_check(rref_matrix)

    if check == 0:

        return r"\text{\ No Solution}"
    
    elif check == 1:

        return r"\text{\ Infinitely Many Solutions}"


    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"
    latex_str += f"x = {rref_matrix[0][-1]} \\\\"
    latex_str += f"y = {rref_matrix[1][-1]} \\\\"
    latex_str += f"z = {rref_matrix[2][-1]}"
    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def matrix_to_tex4x5(matrix):

    # Converts matrix to system of equations

    a, b, c, d, e = matrix[0]
    f, g, h, i, j = matrix[1]
    k, l, m, n, o = matrix[2]
    p, q, r, s, t = matrix[3]

    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"

    # Row 1
    latex_str += (
        f"{format_coef4x5(0, 0, 0, a, 'x')}"
        f"{format_coef4x5(a, 0, 0, b, 'y')}"
        f"{format_coef4x5(a, b, 0, c, 'z')}"
        f"{format_coef4x5(a, b, c, d, 'w')}"
        f"= {e} \\\\"
    )

    # Row 2
    latex_str += (
        f"{format_coef4x5(0, 0, 0, f, 'x')}"
        f"{format_coef4x5(f, 0, 0, g, 'y')}"
        f"{format_coef4x5(f, g, 0, h, 'z')}"
        f"{format_coef4x5(f, g, h, i, 'w')}"
        f"= {j} \\\\"
    )

    # Row 3
    latex_str += (
        f"{format_coef4x5(0, 0, 0, k, 'x')}"
        f"{format_coef4x5(k, 0, 0, l, 'y')}"
        f"{format_coef4x5(k, l, 0, m, 'z')}"
        f"{format_coef4x5(k, l, m, n, 'w')}"
        f"= {o} \\\\"
    )

    # Row 4
    latex_str += (
        f"{format_coef4x5(0, 0, 0, p, 'x')}"
        f"{format_coef4x5(p, 0, 0, q, 'y')}"
        f"{format_coef4x5(p, q, 0, r, 'z')}"
        f"{format_coef4x5(p, q, r, s, 'w')}"
        f"= {t}"
    )

    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def solution_to_tex4x5(rref_matrix):

    check = singular_check(rref_matrix)

    if check == 0:

        return r"\text{\ No Solution}"
    
    elif check == 1:

        return r"\text{\ Infinitely Many Solutions}"


    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"
    latex_str += f"x = {rref_matrix[0][-1]} \\\\"
    latex_str += f"y = {rref_matrix[1][-1]} \\\\"
    latex_str += f"z = {rref_matrix[2][-1]} \\\\"
    latex_str += f"w = {rref_matrix[3][-1]}"
    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def arb_matrix_to_tex(matrix):

    # Builds LaTeX string

    equations, variables = matrix.shape

    latex_str = r"\begin{cases}"
    latex_str += r"\begin{aligned}"
    for i in range(equations):
        latex_str += f"{arb_format_coef(1, matrix[i][0],'x_1')}"
        for j in range(1, variables-1):
            latex_str += f"{arb_format_coef(matrix[i][j-1], matrix[i][j],f'x_{{{j+1}}}')}"
        latex_str += f"= {matrix[i][-1]} \\\\"
    latex_str += r"\end{aligned}"
    latex_str += r"\end{cases}"

    return latex_str

def problem_generator2x3(prob_num = 6, difficulty = "Beginner", singular = "Never"):

    problems = []
    matrices = []

    for _ in range(prob_num):

        if singular == "Never":

            if difficulty == "Beginner":
                M = beginner_2x3(-5, 5)
            elif difficulty == "Intermediate":
                M = beginner_2x3(-12, 12)
            else:
                M = advanced_2x3(-20, 20)

        elif singular == "Always":

            if difficulty == "Beginner":
                M = beginner_2x3_sing(-5, 5)
            elif difficulty == "Intermediate":
                M = beginner_2x3_sing(-12, 12)
            else:
                M = advanced_2x3_sing(-20, 20)

        else:
            # Probability a singular system is generated is 0.2

            if np.random.rand() <= 0.2:

                if difficulty == "Beginner":
                    M = beginner_2x3_sing(-5, 5)
                elif difficulty == "Intermediate":
                    M = beginner_2x3_sing(-12, 12)
                else:
                    M = advanced_2x3_sing(-20, 20)

            else:

                if difficulty == "Beginner":
                    M = beginner_2x3(-5, 5)
                elif difficulty == "Intermediate":
                    M = beginner_2x3(-12, 12)
                else:
                    M = advanced_2x3(-20, 20)

        latex = matrix_to_tex2x3(M)
        problems.append(latex)
        matrices.append(M)

    return (problems,matrices)

def problem_generator3x4(prob_num = 6, difficulty = "Beginner", singular = "Never"):

    problems = []
    matrices = []

    for _ in range(prob_num):

        if singular == "Never":

            if difficulty == "Beginner":
                M = beginner_3x4(-5, 5)
            elif difficulty == "Intermediate":
                M = beginner_3x4(-12, 12)
            else:
                M = advanced_3x4(-20, 20)

        elif singular == "Always":

            if difficulty == "Beginner":
                M = beginner_3x4_sing(-5, 5)
            elif difficulty == "Intermediate":
                M = beginner_3x4_sing(-12, 12)
            else:
                M = advanced_3x4_sing(-20, 20)

        else:
            # Probability a singular system is generated is 0.2

            if np.random.rand() <= 0.2:

                if difficulty == "Beginner":
                    M = beginner_3x4_sing(-5, 5)
                elif difficulty == "Intermediate":
                    M = beginner_3x4_sing(-12, 12)
                else:
                    M = advanced_3x4_sing(-20, 20)

            else:

                if difficulty == "Beginner":
                    M = beginner_3x4(-5, 5)
                elif difficulty == "Intermediate":
                    M = beginner_3x4(-12, 12)
                else:
                    M = advanced_3x4(-20, 20)

        latex = matrix_to_tex3x4(M)
        problems.append(latex)
        matrices.append(M)

    return (problems,matrices)

def problem_generator4x5(prob_num = 6, difficulty = "Beginner", singular = "Never"):

    problems = []
    matrices = []

    for _ in range(prob_num):

        if singular == "Never":

            if difficulty == "Beginner":
                M = beginner_4x5(-5, 5)
            elif difficulty == "Intermediate":
                M = beginner_4x5(-12, 12)
            else:
                M = advanced_4x5(-20, 20)

        elif singular == "Always":

            if difficulty == "Beginner":
                M = beginner_4x5_sing(-5, 5)
            elif difficulty == "Intermediate":
                M = beginner_4x5_sing(-12, 12)
            else:
                M = advanced_4x5_sing(-20, 20)

        else:
            # Probability a singular system is generated is 0.2

            if np.random.rand() <= 0.2:

                if difficulty == "Beginner":
                    M = beginner_4x5_sing(-5, 5)
                elif difficulty == "Intermediate":
                    M = beginner_4x5_sing(-12, 12)
                else:
                    M = advanced_4x5_sing(-20, 20)

            else:

                if difficulty == "Beginner":
                    M = beginner_4x5(-5, 5)
                elif difficulty == "Intermediate":
                    M = beginner_4x5(-12, 12)
                else:
                    M = advanced_4x5(-20, 20)

        latex = matrix_to_tex4x5(M)
        problems.append(latex)
        matrices.append(M)

    return (problems, matrices)

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

#display(Math(matrix_to_tex(small_nice_2x3(-10,10))))


#def nice_matrix(m,n):



# %%

if "problems" not in st.session_state:
    st.session_state.problems = None

if "matrices" not in st.session_state:
    st.session_state.matrices = None

if "show_solutions" not in st.session_state:
    st.session_state.show_solutions = False

difficulty = st.selectbox(
    "Mode:",
    ["Beginner", "Intermediate", "Advanced"]
)

system_size = st.selectbox(
    "System size:",
    [
        "2 variables (2×3)",
        "3 variables (3×4)",
        "4 variables (4×5)"
    ]
)

if system_size == "2 variables (2×3)":
    n_vars = 2
    generator = problem_generator2x3
    formatter = matrix_to_tex2x3

elif system_size == "3 variables (3×4)":
    n_vars = 3
    generator = problem_generator3x4
    formatter = matrix_to_tex3x4

elif system_size == "4 variables (4×5)":
    n_vars = 4
    generator = problem_generator4x5
    formatter = matrix_to_tex4x5

singular = st.selectbox(
    "Allow infinite/no solution?:",
    ["Never", "Sometimes", "Always"]
)

generate = st.button("Generate system")

# %%
def show_problems_in_columns(problems):
    # Start LaTeX string
    latex_str = r"\displaystyle \begin{array}{c c}"  # two centered columns
    
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


def solution_to_tex(matrices, sol_formatter):

    solutions = []
    for M in matrices:
        rref = system_solver(M)
        solutions.append(sol_formatter(rref))
    return solutions

def show_solutions_in_columns(solutions):

    latex_str = r"\displaystyle \begin{array}{c c}"
    
    for i in range(0, len(solutions), 2):
        row = solutions[i:i+2]
        if len(row) == 1:
            row_str = f"\\text{{{i+1}.)}} {row[0]} & ~ "
        else:
            row_str = f"\\text{{{i+1}.)}} {row[0]} & \\text{{{i+2}.)}} {row[1]}"
        latex_str += row_str + r" \\[40pt]"
    
    latex_str += r"\end{array}"
    st.latex(latex_str)

if generate:
    problems, matrices = generator(
        prob_num=6,
        difficulty=difficulty,
        singular=singular
    )

    # Store BOTH so they persist across reruns
    st.session_state.problems = problems
    st.session_state.matrices = matrices

    # Reset solution visibility whenever we generate new problems
    st.session_state.show_solutions = False

if st.session_state.problems is not None:
    show_problems_in_columns(st.session_state.problems)

if st.session_state.problems is not None:
    if st.button("Show solutions"):
        st.session_state.show_solutions = True

if n_vars == 2:
    sol_formatter = solution_to_tex2x3
elif n_vars == 3:
    sol_formatter = solution_to_tex3x4
else:
    sol_formatter = solution_to_tex4x5

if st.session_state.show_solutions:
    st.markdown("### Solutions")

    solutions = solution_to_tex(
        st.session_state.matrices,
        sol_formatter
    )

    show_solutions_in_columns(solutions)


