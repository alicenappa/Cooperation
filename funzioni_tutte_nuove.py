import numpy as np
import matplotlib.pyplot as plt
import random
import math
import ternary
#from funzioni_tutte_nuove import *
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

#GENERALI

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
    
    return float(np.clip(overlap, 0.0, 1.0)) #clip forza tra 0 e 1 è un controllo in piu 

def calculate_total_overlap(population):
    'calculate total overlap of the population' 
    somma = 0
    N = len(population)
    for i in range(N): 
        for j in range(i+1, N): #ciclo su tutte le coppie distinte
            somma += calculate_overlap(population[i],population[j])
    overlap = (2*somma) / (N*(N-1)) #faccio la media: 2/denom è esattemante il numero di coppie distinte
    return overlap


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
    
    cluster_labels = fcluster(linkage_matrix, t=1-similarity_threshold, criterion='distance')
    return cluster_labels

#BASE SENZA OPEN MINDNESS

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


def evolve_population_1( pop_iniz , time , eps, alpha): #ritorna la pop finale, lo storico delle popolazioni, la lista di p_agree e la lista di overlaps
    pop_finale =  np.copy(pop_iniz)
    storico = {} #elenco di popolazioni nel tempo
    ps_agree = []
    overlaps = []
    for t in range(time): 
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)  #non controlla che i1 e i1 non siano lo stesso?
        i1 , i2 = pop_finale[i1_index] , pop_finale[i2_index]

        i1_new , o_ij , p_agree = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
        ps_agree.append(p_agree)
        overlaps.append(o_ij)
    return pop_finale   , storico , ps_agree, overlaps


def evolve_population_2(pop_iniz , time , eps , alpha): #ritorna solo la pop finale

    pop_finale = np.copy(pop_iniz)

    for t in range(time):
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)

        i1 = pop_finale[i1_index]
        i2 = pop_finale[i2_index]

        i1_new, _, _ = interact_individuals(i1, i2, eps, alpha)
        pop_finale[i1_index] = i1_new

    return pop_finale


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
    


# INTERACTION WITH EXTERNAL INFORMATION
def external_info(K, a): #ritorna la matrice degli agenti ext
    assert 0 <= a <= 1, "a deve essere compreso tra 0 e 1"
    off_diagonal_value = (1 - a) / (K - 1)
    M = np.full((K, K),off_diagonal_value)
    np.fill_diagonal(M, a)
    return M


def interact_with_info_proportional(i1, eps, alpha, I, PI):          #interagisce con la source proporzionalmente all'overlap
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        overlaps = np.array([calculate_overlap(i1, src) for src in sources])
        s = overlaps.sum()
        if s == 0:     #controllo che la somma non sia zero per evitare divisione per zero
            print("Warning: All overlaps are zero. Choosing a random source.")
            idx = np.random.randint(0, len(sources))
        else:
            probs = overlaps / s
            idx = np.random.choice(len(sources), p=probs)
        i1_updated , o , p_agree= interact_individuals(i1, i2=I[idx], eps=eps, alpha=alpha)
        return i1_updated
    else: 
        return i1
    
def evolve_population_with_info_proportional(pop, time ,eps, alpha , I , PI):   
    storico = {}
    N = pop.shape[0]
    pop_evoluta = np.copy(pop)
    for t in range(time):
        i1_index = np.random.choice(N)#, size=1, replace=False)     
        i1  = pop_evoluta[i1_index]#[0]  #credo si auguale a  pop_evoluta[i1_index]
        pop_evoluta[i1_index] = interact_with_info_proportional(i1 , eps, alpha , I , PI ) 
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico
    
def evolve_population_with_prop_info_and_peer(pop, time ,eps, alpha , I , PI): #versione alberto bracci: utenti prima interagiscono tra loro e poi con ext source
    storico = {}
    N = pop.shape[0]
    pop_evoluta = np.copy(pop)
    for t in range(time):
        
        i1_index = random.randint(0, len(pop) - 1)
        i2_index = random.randint(0, len(pop) - 1)
        i1 , i2 = pop_evoluta[ i1_index ] , pop_evoluta[i2_index] 
        i_1 , o ,p_agree = interact_individuals(i1, i2 , eps, alpha)
        pop_evoluta[i1_index] = interact_with_info_proportional(i_1 , eps, alpha , I , PI ) 
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico

#OPEN MINDNESS VERSION

def interact_individuals_p_om_fissata(i1, i2, eps, alpha, p_om):
    K = len(i1)
    o_ij = calculate_overlap(i1, i2)

    if random.random() < p_om:
        p_agree = 0.5
    else:
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
    
    return i1_final, 1-o_ij

def evolve_population_p_om_fissata( pop_iniz , time , eps, alpha , p_om): #no self consistece -> p_om data da fuori fissa uguale per tutti
    pop_finale =  np.copy(pop_iniz)
    storico = {}
    storico_interaction = {}
    for t in range(time): 
        #print('TEMPO T = ' , t)
        #estraggo due individui casuali
        i1_index = random.randint(0, len(pop_iniz) - 1)
        i2_index = random.randint(0, len(pop_iniz) - 1)
        i1 , i2 = pop_finale[i1_index] , pop_finale[i2_index]
        #print('individuo 1 numero' , i1_index , 'interagisce con individuo 2 numero' , i2_index)
        if i1_index not in storico_interaction:
            storico_interaction[i1_index] = {} #sto creando il dizionario per i1
        storico_interaction[i1_index].setdefault(t, []) #ti assicuri che, per quell’individuo e per quel tempo t, esista una lista pronta a ricevere dati; se la chiave t non c’è, la crea e ci mette come valore una lista vuota, se invece c’è già non la tocca
        i1_new , distance = interact_individuals_p_om_fissata(i1, i2, eps, alpha , p_om)
        storico_interaction[i1_index][t].append([distance , i2_index])
        # print('storico interaction',storico_interaction)
        pop_finale[i1_index] = i1_new
        storico[t] = np.copy(pop_finale)
    return pop_finale   , storico  , storico_interaction

#OPEN MINDNESS + EXT INFO

def interact_with_pol_info_p_om(i1, eps, alpha, I, PI, p_om):          #interagisce con la source proporzionalmente all'overlap (in realtà nel paper la prob di interagire con la source è prop al vettore di preferenze dell'individuo... da discutere)
    if random.random() < PI:
        K = len(i1)
        sources = I[:K]  # Select the first K sources from I
        overlaps = np.array([calculate_overlap(i1, src) for src in sources])
        s = overlaps.sum()
        if s == 0:     #controllo che la somma non sia zero per evitare divisione per zero
            print("Warning: All overlaps are zero. Choosing a random source.")
            idx = np.random.randint(0, len(sources))
        else:
            probs = overlaps / s
            idx = np.random.choice(len(sources), p=probs)
        i1_updated , o = interact_individuals_p_om_fissata(i1, i2=I[idx], eps=eps, alpha=alpha, p_om=p_om)
        return i1_updated
    else: 
        return i1
    
def interact_with_mild_info(i1, eps, alpha, p_om):
    K = len(i1)
    mild_info=np.array([1/K]*K)
    i1_updated , o  = interact_individuals_p_om_fissata(i1, i2=mild_info, eps=eps, alpha=alpha, p_om=p_om)
    return i1_updated

def interact_with_info(i1, eps, alpha, I, PI, m, p_om):
    if random.random() < m:
        return  interact_with_mild_info(i1, eps, alpha, p_om)
    else:
        return interact_with_pol_info_p_om(i1, eps, alpha, I, PI, p_om)
    
def evolve_population_with_pol_info_mild_info_and_peer(pop, time ,eps, alpha , I , PI, m, p_om): #camilla: versione alberto bracci: utenti prima interagiscono tra loro e poi con ext source
    storico = {}
    N = pop.shape[0]
    pop_evoluta = np.copy(pop)
    for t in range(time):  
        i1_index = random.randint(0, len(pop) - 1)
        i2_index = random.randint(0, len(pop) - 1)
        i1 , i2 = pop_evoluta[ i1_index ] , pop_evoluta[i2_index] 
        i_1 , d = interact_individuals_p_om_fissata(i1, i2 , eps, alpha, p_om)
        pop_evoluta[i1_index] = interact_with_info(i_1 , eps, alpha , I , PI, m, p_om) 
        storico[t] = np.copy(pop_evoluta)
    return pop_evoluta , storico


