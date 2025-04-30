
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import ternary
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
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
    return overlap

def calculate_total_overlap(population):
    'calculate total overlap of the population' 
    somma = 0
    N = len(population)
    for i in range(N):
        for j in range(i+1, N):
            somma += calculate_overlap(population[i],population[j])
    overlap = (2*somma) / (N*(N-1))
    return overlap


def evolve_population( pop_iniz , time , eps, alpha):
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    ps_agree = []
    overlaps = []
    for t in range(time): 
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 = pop_finale[ i1_index ] , pop_finale[i2_index]

        i1_new,p_agree , o_ij = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
        ps_agree.append(p_agree)
        overlaps.append(o_ij)
    return pop_finale   , storico , ps_agree, overlaps





def generate_population(K , N,entropy_treshold):
    'generates a population of N individuals of K components with that entropy trashold'
    population = []
    while len(population) != N:
        vec = np.random.dirichlet(np.ones(K))         
        entropia = entropy(vec)
        if entropia > entropy_treshold:
            if random.random() < 0.9:
                continue
            else:
                population.append(vec) 
        else: 
            population.append(vec)          
    return population


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
            
            
        if  counter > 50:
            break 
      #  print(sum(x))
    return x     
def interact_individuals(i1, i2, eps, alpha):
#print('vector i1' , i1 , 'interacts with vector i2' , i2)
    K = len(i1)  # Numero totale di elementi
    o_ij = calculate_overlap(i1, i2)
    signo = random.choice([-1, +1])
    p_agree = min(1, max(0, o_ij + eps*signo))
    l = random.choice(range(K))
    # Calcolo della variazione delta
    if abs(i2[l] - i1[l]) > alpha:
        delta = alpha * np.sign(i2[l] - i1[l])
        
    else:
        delta = 0.5 * (i2[l] - i1[l])
    
    i1_new = i1.copy()
    resto = 0 
    # Modifica dell'elemento selezionato
    if random.random() < p_agree:
        i1_new[l] += delta
        beta = delta  
    else:
        i1_new[l] -= delta
        beta = -delta 
        
    if i1_new[l] < 0: 
        resto += i1_new[l]
        i1_new[l] = 0
        
    if i1_new[l] >= 1:
       
        i1_new = np.zeros(len(i1_new))
        i1_new[l] = 1
        return i1_new
    
    i1_last = update(np.array(i1_new) , l , beta - resto)
    
#    print(sum(i1_last))
    return np.array(i1_last) 


def evolve_population( pop_iniz , time , eps, alpha):
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    for t in range(time): 
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 = pop_finale[ i1_index ] , pop_finale[i2_index]

        i1_new = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
    return pop_finale   , storico 


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
    indices_over_one = np.argwhere(similarity_matrix > 1)

    # Stampa gli indici e i relativi valori
    for i, j in indices_over_one:
        similarity_matrix[i][j] = 1 
    distance_matrix = 1 - similarity_matrix
    condensed_distance = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distance, method='complete')
    
    distance_threshold = 1 - similarity_threshold
    K = len(population[0])
    cluster_labels = fcluster(linkage_matrix, t = similarity_threshold , criterion='distance')

    return cluster_labels


# INTERACTION WITH EXTERNAL INFORMATION 
def external_info(K, a):
    assert 0 <= a <= 1, "a deve essere compreso tra 0 e 1"
    off_diagonal_value = (1 - a) / (K - 1)
    M = np.full((K, K),off_diagonal_value)
    np.fill_diagonal(M, a)
    return M


def interact_with_info(i1, eps, alpha, I, PI): 
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        overlaps = [calculate_overlap(i1, src) for src in sources]
        max_index = overlaps.index(max(overlaps))  
        i1_updated = interact_individuals(i1, i2=I[max_index], eps=eps, alpha=alpha)
        return i1_updated
    else: 
        return i1


    
def interact_with_random_info(i1, eps, alpha, I, PI): 
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        i2 =  random.randint(0, K-1)  
        i1_updated = interact_individuals(i1, i2, eps=eps, alpha=alpha)
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
        pop_evoluta[i3_index] = interact_individuals(i2, i3 , eps, alpha)
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico


