# MIT 6.034 Lab 7: Support Vector Machines
# Written by Jessica Noss (jmn) and 6.034 staff

from svm_data import *
import math

# Vector math
def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    return sum([x * y for (x,y) in zip(u,v)])


def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return math.sqrt(sum([x*x for x in v]))

# Equation 1
def positiveness(svm, point):
    "Computes the expression (w dot x + b) for the given point"
    return dot_product(svm.w,point.coords) + svm.b

def classify(svm, point):
    """Uses given SVM to classify a Point.  Assumes that point's true
    classification is unknown.  Returns +1 or -1, or 0 if point is on boundary"""
    pos = positiveness(svm,point)
    if pos > 0:
        return 1
    elif pos < 0:
        return -1
    else:
        return 0

# Equation 2
def margin_width(svm):
    "Calculate margin width based on current boundary."
    return 2/norm(svm.w)

# Equation 3
def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""
    bad_points = set()

    #Check support_vector on correct boundary
    bad_points = bad_points.union(filter(lambda s_v: s_v.classification != positiveness(svm,s_v),svm.support_vectors))

    for training_point in svm.training_points:
        if positiveness(svm,training_point) <1 and positiveness(svm,training_point) > -1:
            bad_points.add(training_point)
            
    return bad_points


# Equations 4, 5
def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM h as support vectors assigned, and that all training
    points have alpha values assigned."""
    bad_points = set()

    bad_points  = bad_points.union(filter(lambda s_v:  not(s_v.alpha > 0), svm.support_vectors))

    bad_points = bad_points.union(filter(lambda t_p: t_p.alpha != 0 and t_p not in svm.support_vectors, svm.training_points))

    return bad_points

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False.  Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""
    
    if sum([t_p.alpha*classify(svm,t_p) for t_p in svm.training_points]) != 0:
        return False
    if svm.w != reduce(vector_add,[scalar_mult(t_p.alpha*classify(svm,t_p),t_p.coords) for t_p in svm.training_points]):
        return False
    return True


# Classification accuracy
def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    return set(filter(lambda t_p: classify(svm,t_p) != t_p.classification , svm.training_points))

# Training
def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b.  Return the updated SVM."""
    # update support_vectors
    support_vectors = filter(lambda t_p: t_p.alpha >0, svm.training_points)
    svm.support_vectors = support_vectors

    # try with t_p instead of t_p
    svm.w = reduce(vector_add,[scalar_mult(t_p.classification*t_p.alpha,t_p.coords) for t_p in svm.training_points])

    
    pos_support_vectors = filter(lambda s_p: s_p.classification == 1, svm.support_vectors)
    neg_support_vectors = filter(lambda s_p: s_p.classification == -1, svm.support_vectors)

    pos_support_vectors_max_b = max([s_p.classification - dot_product(svm.w,s_p.coords) for s_p in pos_support_vectors])
    neg_support_vectors_min_b = min([s_p.classification - dot_product(svm.w,s_p.coords) for s_p in neg_support_vectors])
    svm.b = .5*(pos_support_vectors_max_b + neg_support_vectors_min_b)
    
    return svm

# Multiple choice
ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A','D']
ANSWER_6 = ['A','B','D']
ANSWER_7 = ['A','B','D']
ANSWER_8 = []
ANSWER_9 = ['A','B','D']
ANSWER_10 = ['A','B','D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8]
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
