"""

    Program do modelowania epidemii (modelowanych za pomocą modelu SIS)
    na grafach losowych wygenerowanych za pomocą modelu Barabasi-Albert
    
"""

import matplotlib.pyplot as plt
import random

def main():
    # my code here
    generate_graph(4,2,6);


"""
    m_0 - liczba wierzcholkow w poczatkowym grafie pelnym
    m - ile polaczen ma utworzyc nowo dodany wierzcholek
    t - zadana chwila czasowa

"""
def generate_graph(m_0,m,t):
    print ("Generowanie grafu dla: \n\tm_0 =",m_0,"\n\tm = ",m,"\n\tt =",t)
    # stworz wstepny graf (graf pełny - całkowicie połączony klaster węzłów):
    graph = dict()
    for m_i in range(0,m_0):
        l = [x+1 for x in range(m_0)] # dodaj wszystkie wezly (indeksowanie od 1)
        del l[m_i] # usun siebie
        graph[m_i + 1] = l
    
    # zapelniaj go:
    for t_i in range(0,t):
        pref_addition(graph,m) # dodaj nowy wezel, ktory stworzy m polaczen (na koniec slownika)   
    
    print_graph(graph)
    return
    

def pref_addition(g,m):
    num_node = len(g) + 1
    
    """
    wybierz wezly, do ktorych polaczy sie nowy wezel 
    """
    total_p = 0
    for v in g.values():
        total_p += len(v) 
    marked = []
    while len(marked) < m:
        p = random.randint(0,total_p)
        cum_p = 0
        for k, v in g.items():
            cum_p += len(v)
            if (not k in marked) and (p <= cum_p):
                marked.append(k)
                # updatuj graf (do wczesniejszego wezla dodaj wskazanie na nowy):
                g[k].append(num_node)
                break 
    # updatuj graf (dodaj nowy wezel):
    new_node = {num_node: marked}
    g.update(new_node)
    return

 
# Utilities:
def print_graph(g):
    print ("Graf o",len(g),"wierzcholkach:")
    for k, v in g.items():
        print (k, ':', v)
    return
    
def plot_graph(g):
    return


if __name__ == "__main__":
    main()
