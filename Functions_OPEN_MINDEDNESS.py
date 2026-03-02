
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import ternary
from Functions_sirbu_loreto import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform


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

def calculate_w(ind, storico_interaction , mu, t ):
    den = mu
    if t < mu:
        mu = t -1      

    somma = 0

    for t_prime in range( t - mu - 1 , t):  # range esclude il valore finale quindi metto t
        if t_prime not in storico_interaction[ind]:
            continue

        for int_type, dist, i2 , w  in storico_interaction[ind][t_prime]:
            if int_type == +1:
                somma += dist
                continue
            if int_type == -1:
                somma += 1 - dist
                continue
            if int_type == 'om':
                somma += 1
                continue

    return somma / den



def calculate_total_overlap(population):
    'calculate total overlap of the population' 
    somma = 0
    N = len(population)
    for i in range(N):
        for j in range(i+1, N):
            somma += calculate_overlap(population[i],population[j])
    overlap = (2*somma) / (N*(N-1))
    return overlap



def interact_individuals_om(i1, i1_index, i2, eps, alpha , storico_interaction ,t , mu):
    #parameter definition 
    K = len(i1)  
    l = random.choice(range(K))
    o_ij = calculate_overlap(i1, i2)
    d = 1 - o_ij
   
    w = calculate_w(i1_index , storico_interaction , mu , t) 
   
    if d < 0.5:
        f_d = 0.5 + d
    if d >= 0.5: 
        f_d = 3/2 - d
   
        
    p_open_minded = w * f_d
    
    i1_new = i1.copy()
    resto = 0  
    # Calcolo della variazione delta
    if abs(i2[l] - i1[l]) > alpha:
        delta = alpha * np.sign(i2[l] - i1[l])
        
    else:
        delta = 0.5 * (i2[l] - i1[l])
    
    # DEFINE TWO KINDS OF INTERACTION POSSIBLE: OM OR NORMAL 
    if random.random() < p_open_minded:   #open minded interaction
        interaction_type = 'om'
        p_agree = 0.5
        if random.random() < p_agree:
            i1_new[l] += delta
            beta = delta  
        else:
            i1_new[l] -= delta
            beta = -delta
        
    else:      #NORMAL INTERACTION
        signo = random.choice([-1, +1])
        p_agree = min(1, max(0, o_ij + eps*signo))
        if random.random() < p_agree:
            interaction_type = +1
            i1_new[l] += delta
            beta = delta  
        else:
            interaction_type = -1
            i1_new[l] -= delta
            beta = -delta 
               
    # once calculate the variation of the vector, let's vary the vector 
      
    if i1_new[l] < 0: 
        resto += i1_new[l]
        i1_new[l] = 0
        
    if i1_new[l] >= 1:
       
        i1_new = np.zeros(len(i1_new))
        i1_new[l] = 1
        return i1_new, interaction_type,d  , w # se è più di 1 mettiamo vettore unitario e returniamo direttametne senza chiamare funzione di update 
    
    i1_last = update(np.array(i1_new) , l , beta - resto)
    
    
    return np.array(i1_last), interaction_type , d , w



def evolve_population( pop_iniz , time , eps, alpha ,mu):
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    storico_interaction = {}
    for t in range(time): 
        #print('TEMPO T = ' , t)
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 = pop_finale[ i1_index ] , pop_finale[i2_index]
       # print('individuo 1 numero' , i1_index , 'interagisce con individuo 2 numero' , i2_index)
        if i1_index not in storico_interaction:
            storico_interaction[i1_index] = {}

        # più avanti, subito prima di appendere il risultato:
        storico_interaction[i1_index].setdefault(t, [])
        # ora posso fare
        i1_new , interaction , distance , w = interact_individuals_om(i1, i1_index, i2, eps, alpha , storico_interaction , t , mu)
        storico_interaction[i1_index][t].append([interaction, distance , i2_index ,w ])
        
       # print('storico interaction',storico_interaction)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
    return pop_finale   , storico  , storico_interaction





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
            
            
        if  counter > 50:
            break 
      #  print(sum(x))
    return x     

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
        i1_updated , o , p = interact_individuals(i1, i2=I[max_index], eps=eps, alpha=alpha)
        return i1_updated
    return i1


    
def interact_with_random_info(i1, eps, alpha, I, PI): 
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        i2 =  random.randint(0, K-1)  
        i1_updated , o , p= interact_individuals(i1, i2, eps=eps, alpha=alpha)
        return i1_updated
    else: 
        return i1
    


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




def evolve_population_with_info_and_peer_openm( pop_iniz , time , eps, alpha , I , PI , mu):
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    storico_interaction = {}
    for t in range(time): 
        #print('TEMPO T = ' , t)
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i3_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 , i3 = pop_finale[ i1_index ] , pop_finale[i2_index] , pop_finale[i3_index]
       # print('individuo 1 numero' , i1_index , 'interagisce con individuo 2 numero' , i2_index)
        if i1_index not in storico_interaction:
            storico_interaction[i1_index] = {}

       
        storico_interaction[i1_index].setdefault(t, [])
        i1_new , interaction , distance,w  = interact_individuals_om(i1, i1_index, i2, eps, alpha , storico_interaction , t , mu)
        storico_interaction[i1_index][t].append([interaction, distance , i2_index ,w ])
        
       # print('storico interaction',storico_interaction)
        pop_finale[i1_index] = i1_new
        ii = interact_with_info(i3 , eps, alpha , I , PI )
        pop_finale[i3_index] =   ii
        storico[t] = np.copy(pop_finale)
    return pop_finale   , storico  , storico_interaction

