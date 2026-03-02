
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import ternary
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D

def entropy(vec):
    '''calcola l'entropia di shannon di un vettore di probabilità'''
    K = len(vec)
    somma = 0
    for i in vec: 
        if i<0:
            print('!! elemento negativo !!')
        somma += i* math.log2(i + 1e-10)
    return -somma

def calculate_overlap(agent1, agent2):

    'calcola l overlap tra due individui' 

    scalar = np.dot(agent1, agent2)
    norm1 = np.linalg.norm(agent1)
    norm2 = np.linalg.norm(agent2)
    overlap = scalar / (norm1*norm2)
    return float(np.clip(overlap, 0.0, 1.0))

def calculate_total_overlap(population):
    'calculate total overlap of the population' 
    somma = 0
    N = len(population)
    for i in range(N):
        for j in range(i+1, N):
            somma += calculate_overlap(population[i],population[j])
    overlap = (2*somma) / (N*(N-1))
    return overlap


def update(x, l, amount_to_distribute):
    """
    Distributes amount_to_distribute among all indices EXCEPT l.
    Ensures the final sum is exactly 1 and all elements >= 0.
    """
    K = len(x)
    others = [i for i in range(K) if i != l]
    
    # We use a loop to handle 'overflow' if an element hits 0 or 1
    for _ in range(10): 
        if abs(amount_to_distribute) < 1e-15:
            break
            
        if amount_to_distribute > 0:
            eligible = [i for i in others if x[i] > 0]
        else:
            eligible = [i for i in others if x[i] < 1]
            
        if not eligible:
            break
            
        share = amount_to_distribute / len(eligible)
        for i in eligible:
            old_val = x[i]
            x[i] = np.clip(old_val - share, 0, 1)
            amount_to_distribute -= (old_val - x[i])
            
  
    s = np.sum(x)
    if s > 0:
        return x / s
    else:
        return np.ones(K) / K

def interact_individuals(i1, i2, eps, alpha):
    K = len(i1)
    o_ij = calculate_overlap(i1, i2)
    p_agree = min(1, max(0, o_ij + (eps * random.choice([-1, 1]))))
    
    l = random.choice(range(K))
    if abs(i2[l] - i1[l]) > alpha:
        delta = alpha * np.sign(i2[l] - i1[l])
        
    else:
        delta = 0.5 * (i2[l] - i1[l])


    if random.random() >= p_agree:
        delta = -delta
        
    i1_new = i1.copy()
    old_l_val = i1_new[l]
    
    i1_new[l] = np.clip(old_l_val + delta, 0, 1)
    
    actual_delta = i1_new[l] - old_l_val
    
    i1_final = update(i1_new, l, actual_delta)
    
    return i1_final, o_ij, p_agree




def evolve_population_1( pop_iniz , time , eps, alpha):
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    ps_agree = []
    overlaps = []
    for t in range(time): 
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 = pop_finale[ i1_index ] , pop_finale[i2_index]

        i1_new , o_ij , p_agree = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
        ps_agree.append(p_agree)
        overlaps.append(o_ij)
    return pop_finale   , storico , ps_agree, overlaps


def evolve_population_2(pop_iniz , time , eps , alpha):

    pop_finale = np.copy(pop_iniz)

    for t in range(time):
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)

        i1 = pop_finale[i1_index]
        i2 = pop_finale[i2_index]

        i1_new, _, _ = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new

    return pop_finale




def generate_population(K, N, entropy_threshold):
    '''
    Generates a population of N individuals with a given entropy threshold
    '''
    population = []
    while len(population) < N:
        # Generate a point on the K-simplex
        vec = np.random.dirichlet(np.ones(K))         
        S = entropy(vec)
        
        if S > entropy_threshold:
            
            if random.random() < 1:    #here put 0.9 if you want 
                continue  # Discard and try again
        
        # If entropy is good OR we passed the 10% random check
        population.append(vec) 
                 
    return np.array(population)

def plot_simplesso(population):
    # Creazione del grafico ternario
    fig, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=2.0)  # Disegna i bordi del triangolo
    tax.gridlines(multiple=0.2, color="gray", linewidth=0.5)  # Aggiunge una griglia
    tax.scatter(population, marker="o", color="blue", s=10, alpha=0.7)
    tax.right_corner_label("$p_1$", fontsize=12)
    tax.top_corner_label("$p_2$", fontsize=12)
    tax.left_corner_label("$p_3$", fontsize=12)
    tax.show()



def plot_simplesso_with_ax(population, ax, title):
    
    scale = 1.0
    figure, tax = ternary.figure(scale=scale, ax=ax)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.2, color="gray", linewidth=0.5)
    
   
    tax.scatter(population, marker="o", color="blue", s=10, alpha=0.7)

   
    #tax.right_corner_label("$p_1$", fontsize=12)
    #tax.top_corner_label("$p_2$", fontsize=12)
    #tax.left_corner_label("$p_3$", fontsize=12)
    tax.set_title(title, fontsize=10)

''' 
def update(x , l , delta):
   # print('update x l delta ' ,  x , l , delta)
    K = len(x)
    indices = np.arange(len(x))
    indices = np.delete(indices,l)
    tolleranza = 1e-6 
    counter = 0 
    while abs(sum(x) - 1) > tolleranza or min(x) < 0 or max(x) > 1 :
        counter += 1 
      #  print('entro nel ciclo')
        for i in indices:
            x[i] -= delta /(len(indices))
        #    print(x[i] , 'updated')
     #   print(indices,'indices')
        negative_indices = [i for i, j in enumerate(x) if j < 0]
    #    print(negative_indices , 'indici negativi')
        negative_elements = [j for j in x if j < 0]
   #     print(negative_elements, 'elementi negativi')
        if len(negative_indices) > 0:
            delta = abs(sum(negative_elements))
            for i in negative_indices:
                x[i] = 0 
            indices = np.delete(indices, np.where(np.isin(indices, negative_indices))[0])
            
            
        if  counter > 500:
            break 
      #  print(sum(x))
    return x     
'''







'''
def evolve_population( pop_iniz , time , eps, alpha):
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    for t in range(time): 
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 = pop_finale[ i1_index ] , pop_finale[i2_index]

        i1_new , o = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
    return pop_finale   , storico 
'''

def plot_user_trajectory_from_dict(storico, index_user, ax):
    """
    Traccia la traiettoria di un singolo individuo nel simplesso usando il dizionario storico.

    Parametri:
    - storico: dizionario {t: popolazione_t}, dove popolazione_t è una lista di array (N x 3)
    - index_user: indice dell'individuo da tracciare
    - ax: asse su cui disegnare il simplesso
    - title: titolo del grafico
    """
    scale = 1.0
    figure, tax = ternary.figure(scale=scale, ax=ax)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.2, color="gray", linewidth=0.5)
    ultimo_tempo = max(storico.keys())  # Ultimo istante
    tax.scatter(storico[ultimo_tempo], marker="o", color="blue", s=10, alpha=0.7)
    user_trajectory = [storico[t][index_user] for t in sorted(storico.keys())]
    tax.plot(user_trajectory, linewidth=2, linestyle="-", color="red", markersize=5, alpha=0.8)
    tax.scatter([user_trajectory[0]], marker="o", color="green", s=40, label="Inizio")  # Punto iniziale
    tax.scatter([user_trajectory[-1]], marker="o", color="black", s=40, label="Fine")  # Punto finale
    
    




# CLUSTERING PART 
def compute_overlap_matrix(population):    
    N = len(population)
    similarity_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            sim = calculate_overlap(population[i], population[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
    return similarity_matrix


def compute_PR(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    return (np.sum(counts)**2) / np.sum(counts**2) if np.sum(counts**2) != 0 else 0


def hierarchical_clustering_K(population, similarity_threshold=0.8):
    similarity_matrix = compute_overlap_matrix(population)
    
    # Ensure similarity is strictly between 0 and 1
    similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)
    
    distance_matrix = 1.0 - similarity_matrix
    
    # squareform will fail if there are NaNs. This is your final safety net.
    if not np.all(np.isfinite(distance_matrix)):
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0)

    condensed_distance = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distance, method='complete')
    
    cluster_labels = fcluster(linkage_matrix, t=similarity_threshold, criterion='distance')
    return cluster_labels


# INTERACTION WITH EXTERNAL INFORMATION 
def external_info(K, a):
    assert 0 <= a <= 1, "a deve essere compreso tra 0 e 1"
    off_diagonal_value = (1 - a) / (K - 1)
    M = np.full((K, K),off_diagonal_value)
    np.fill_diagonal(M, a)
    return M


def interact_with_info(i1, eps, alpha, I, PI):          #interagisce con la source con maggiore overlap 
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        overlaps = [calculate_overlap(i1, src) for src in sources]
        max_index = overlaps.index(max(overlaps))  
        i1_updated , o , p_agree= interact_individuals(i1, i2=I[max_index], eps=eps, alpha=alpha)
        return i1_updated
    else: 
        return i1


    
def interact_with_random_info(i1, eps, alpha, I, PI):           #interagisce con una source casuale 
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        i2 =  random.randint(0, K-1)  
        i1_updated , o , p_agree = interact_individuals(i1, i2, eps=eps, alpha=alpha)
        return i1_updated
    else: 
        return i1
    
def evolve_population_with_info(pop, time ,eps, alpha , I , PI):   
    storico = {}
    N = pop.shape[0]
    pop_evoluta = np.copy(pop)
    for t in range(time):
        i1_index = np.random.choice(N, size=1, replace=False)     
        i1  = pop_evoluta[i1_index][0]    
        pop_evoluta[i1_index] = interact_with_info(i1 , eps, alpha , I , PI ) 
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico

def evolve_population_with_random_info(pop, time ,eps, alpha , I , PI):
    storico = {}
    N = pop.shape[0]
    pop_evoluta = np.copy(pop)
    for t in range(time):
        i1_index = np.random.choice(N, size=1, replace=False)     
        i1  = pop_evoluta[i1_index][0]    
        pop_evoluta[i1_index] = interact_with_random_info(i1 , eps, alpha , I , PI ) 
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico


def evolve_population_with_info_and_peer(pop, time ,eps, alpha , I , PI):
    storico = {}
    N = pop.shape[0]
    pop_evoluta = np.copy(pop)
    for t in range(time):
        
        i1_index = random.randint(0, len(pop) - 1)
        i2_index = random.randint(0, len(pop) - 1)
        i3_index = random.randint(0, len(pop) - 1)
        i1 , i2 , i3 = pop_evoluta[ i1_index ] , pop_evoluta[i2_index], pop_evoluta[i3_index] 
        pop_evoluta[i1_index] = interact_with_info(i1 , eps, alpha , I , PI ) 
        i_3 , o ,p_agree = interact_individuals(i2, i3 , eps, alpha)
        pop_evoluta[i3_index] = i_3
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico

