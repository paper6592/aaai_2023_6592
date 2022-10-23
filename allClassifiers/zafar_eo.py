import numpy as np
from cvxpy import *
import traceback
import sys

# https://github.com/maliha93/Fairness-Analysis-Code/blob/master/Inprocessing/ZafarEO
def sum_array(arr):
    count = np.sum(np.asarray(arr))
    return count

def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and str(type(k)) != "<class 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print (str(type(k)))
            print ("************* ERROR: Input arr does not have integer types")
            return None
        
    in_arr = np.array(in_arr, dtype=int)
    assert(len(in_arr.shape)==1) # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    
    index_dict = {} # value to the column number
    for i in range(0,len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []    
    for i in range(0,len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1 # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict

def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs_to_cov_thresh, cons_type, w):
    """
    get the list of constraints to be fed to the minimizer
    cons_type == 0: means the whole combined misclassification constraint (without FNR or FPR)
    cons_type == 1: FPR constraint
    cons_type == 2: FNR constraint
    cons_type == 4: both FPR as well as FNR constraints
    sensitive_attrs_to_cov_thresh: is a dict like {s: {cov_type: val}}
    s is the sensitive attr
    cov_type is the covariance type. contains the covariance for all misclassifications, FPR and for FNR etc
    """
    constraints = []
    for attr in sensitive_attrs_to_cov_thresh.keys():

        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)
                
        if index_dict is None: # binary attribute, in this case, the attr_arr_transformed is the same as the attr_arr

            s_val_to_total = {ct:{} for ct in [0,1,2]} # constrain type -> sens_attr_val -> total number
            s_val_to_avg = {ct:{} for ct in [0,1,2]}
            cons_sum_dict = {ct:{} for ct in [0,1,2]} # sum of entities (females and males) in constraints are stored here

            for v in set(attr_arr):
                s_val_to_total[0][v] = sum_array(x_control_train[attr] == v)
                s_val_to_total[1][v] = sum_array(np.logical_and(x_control_train[attr] == v, y_train == -1)) # FPR constraint so we only consider the ground truth negative dataset for computing the covariance
                s_val_to_total[2][v] = sum_array(np.logical_and(x_control_train[attr] == v, y_train == +1))


            for ct in [0,1,2]:
                s_val_to_avg[ct][0] = s_val_to_total[ct][1] / float(s_val_to_total[ct][0] + s_val_to_total[ct][1]) # N1/N in our formulation, differs from one constraint type to another
                s_val_to_avg[ct][1] = 1.0 - s_val_to_avg[ct][0] # N0/N

            
            for v in set(attr_arr):

                idx = x_control_train[attr] == v                


                #################################################################
                # #DCCP constraints
                dist_bound_prod = multiply(y_train[idx], x_train[idx] * reshape(w, (w.shape[0], 1))) # y.f(x)
                
                cons_sum_dict[0][v] = sum(  minimum(0, dist_bound_prod) ) * (s_val_to_avg[0][v] / len(x_train)) # avg misclassification distance from boundary
                cons_sum_dict[1][v] = sum(  minimum(0, multiply( (1 - y_train[idx])/2.0, dist_bound_prod) ) ) * (s_val_to_avg[1][v] / sum_array(y_train == -1)) # avg false positive distance from boundary (only operates on the ground truth neg dataset)
                cons_sum_dict[2][v] = sum(  minimum(0, multiply( (1 + y_train[idx])/2.0, dist_bound_prod) ) ) * (s_val_to_avg[2][v] / sum_array(y_train == +1)) # avg false negative distance from boundary
                #################################################################

                
            if cons_type == 4:
                cts = [1,2]
            elif cons_type in [0,1,2]:
                cts = [cons_type]
            
            else:
                raise Exception("Invalid constraint type")


            #################################################################
            #DCCP constraints
            for ct in cts:
                thresh = abs(sensitive_attrs_to_cov_thresh[attr][ct][1] - sensitive_attrs_to_cov_thresh[attr][ct][0])
                constraints.append( cons_sum_dict[ct][1] <= cons_sum_dict[ct][0]  + thresh )
                constraints.append( cons_sum_dict[ct][1] >= cons_sum_dict[ct][0]  - thresh )

            #################################################################


            
        else: # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately
            # need to fill up this part
            raise Exception("Fill the constraint code for categorical sensitive features... Exiting...")
            sys.exit(1)
            

    return constraints

def train_model_disp_mist(x, y, x_control, loss_function, EPS, cons_params=None):
    # cons_type, sensitive_attrs_to_cov_thresh, take_initial_sol, gamma, tau, mu, EPS, cons_type
    """
    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "disparate_mistreatment/synthetic_data_demo/decision_boundary_demo.py"
    ----
    Inputs:
    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    EPS: stopping criteria for the convex solver. check the CVXPY documentation for details. default for CVXPY is 1e-6
    cons_params: is None when we do not want to apply any constraints
    otherwise: cons_params is a dict with keys as follows:
        - cons_type: 
            - 0 for all misclassifications 
            - 1 for FPR
            - 2 for FNR
            - 4 for both FPR and FNR
        - tau: DCCP parameter, controls how much weight to put on the constraints, if the constraints are not satisfied, then increase tau -- default is DCCP val 0.005
        - mu: DCCP parameter, controls the multiplicative factor by which the tau increases in each DCCP iteration -- default is the DCCP val 1.2
        - take_initial_sol: whether the starting point for DCCP should be the solution for the original (unconstrained) classifier -- default value is True
        - sensitive_attrs_to_cov_thresh: covariance threshold for each cons_type, eg, key 1 contains the FPR covariance
    ----
    Outputs:
    w: the learned weight vector for the classifier
    """

    max_iters = 100 # for the convex program
    max_iter_dccp = 50  # for the dccp algo
    num_points, num_features = x.shape
    w = Variable(num_features) # this is the weight vector

    # initialize a random value of w
    np.random.seed(112233)
    w.value = np.random.rand(x.shape[1])

    if cons_params is None: # just train a simple classifier, no fairness constraints
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, cons_params["sensitive_attrs_to_cov_thresh"], cons_params["cons_type"], w)


    if loss_function == "logreg":
        # constructing the logistic loss problem
        loss = sum(  logistic( multiply(-y, x*reshape(w, (w.shape[0], 1))) )  ) / num_points # we are converting y to a diagonal matrix for consistent


    # sometimes, its a good idea to give a starting point to the constrained solver
    # this starting point for us is the solution to the unconstrained optimization problem
    # another option of starting point could be any feasible solution
    if cons_params is not None:
        if cons_params.get("take_initial_sol") is None: # true by default
            take_initial_sol = True
        elif cons_params["take_initial_sol"] == False:
            take_initial_sol = False

        if take_initial_sol == True: # get the initial solution
            p = Problem(Minimize(loss), [])
            try:
                p.solve(verbose=True)
            except SolverError as  se:
                print(se)
                p.solve(solver=SCS, verbose=True)


    # construct the cvxpy problem
    prob = Problem(Minimize(loss), constraints)

    # print "\n\n"
    # print "Problem is DCP (disciplined convex program):", prob.is_dcp()
    # print "Problem is DCCP (disciplined convex-concave program):", is_dccp(prob)

    try:

        tau, mu = 0.005, 1.2 # default dccp parameters, need to be varied per dataset
        if cons_params is not None: # in case we passed these parameters as a part of dccp constraints
            if cons_params.get("tau") is not None: tau = cons_params["tau"]
            if cons_params.get("mu") is not None: mu = cons_params["mu"]

        prob.solve(method='dccp', tau=tau, mu=mu, tau_max=1e10,
            solver=ECOS, verbose=True, 
            feastol=EPS, abstol=EPS, reltol=EPS,feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
            max_iters=max_iters, max_iter=max_iter_dccp)

        
        #assert(prob.status == "Converged" or prob.status == "optimal")
        # print "Optimization done, problem status:", prob.status

    except:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


    # check that the fairness constraint is satisfied
    for f_c in constraints:
        #assert(f_c.value == True) # can comment this out if the solver fails too often, but make sure that the constraints are satisfied empirically. alternatively, consider increasing tau parameter
        pass
        

    w = np.array(w.value).flatten() # flatten converts it to a 1d array
    return w

def trainZafarClassifier_EO(train_dataset):
    x = train_dataset.features[:,:-1]
    y = train_dataset.labels
    y[y == 0] = -1
    z = {train_dataset.protected_attribute_names[0]:train_dataset.protected_attributes.squeeze()}
    cons_type = 4
    tau = 5.0
    mu = 1.2
    loss_function = "logreg"
    eps = 1e-6
    sensitive_attrs_to_cov_thresh = {i: {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}} for i in z}
    cons_params = {"cons_type": cons_type, 
					"tau": tau, 
					"mu": mu, 
					"sensitive_attrs_to_cov_thresh": sensitive_attrs_to_cov_thresh}
    w = train_model_disp_mist(x, y, z, loss_function, eps, cons_params)
    return w