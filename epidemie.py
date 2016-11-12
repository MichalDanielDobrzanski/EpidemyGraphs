"""

    Program do modelowania epidemii (modelowanych za pomocą modelu SIS)
    na grafach losowych wygenerowanych za pomocą modelu Barabasi-Albert
    
    Instalowanie biblioteki do rysowania grafów:
        sudo pip3 install networkx
    
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    # my code here
    m_0 = 4
    m = 2
    t = 100
    
    graph = generate_graph(m_0, m, t)
    print_graph(graph)
    plot_graph(graph,graph_layout='spring')
    plot_graph_degree(graph, m_0, m)
    
    return

"""
    m_0 - liczba wierzcholkow w poczatkowym grafie pelnym
    m - ile polaczen ma utworzyc nowo dodany wierzcholek
    t - zadana chwila czasowa

"""


def generate_graph(m_0, m, t):
    print("Generowanie grafu dla: \n\tm_0 =", m_0, "\n\tm = ", m, "\n\tt =", t)
    
    # stworz wstepny graf (graf pełny - całkowicie połączony klaster węzłów):
    graph = dict()
    for m_i in range(0, m_0):
        l = [x+1 for x in range(m_0)]  # dodaj wszystkie wezly (indeksowanie od 1)
        del l[m_i]  # usun siebie
        graph[m_i + 1] = (0, l)
    # zapelniaj go:
    for t_i in range(0, t):
        pref_addition(graph, m)  # dodaj nowy wezel, ktory stworzy m polaczen (na koniec slownika)
    
    return graph
    

def pref_addition(g, m):
    num_node = len(g) + 1    
    # wybierz wezly, do ktorych polaczy sie nowy wezel 
    total_p = 0
    for v in g.values():
        total_p += len(v) 
    marked = []
    while len(marked) < m:
        p = random.randint(0, total_p)
        cum_p = 0
        for k, v in g.items():
            cum_p += len(v)
            if (not k in marked) and (p <= cum_p):
                marked.append(k)
                # updatuj graf (do wczesniejszego wezla dodaj wskazanie na nowy tworzony):
                g[k][1].append(num_node)
                break 
    # updatuj graf (dodaj nowy wezel wraz z jego polaczeniami):
    new_node = {num_node: (0, marked)}
    g.update(new_node)
    return

 
# Utilities:
def print_graph(g):
    print("Graf o", len(g), "wierzcholkach:")
    for k, v in g.items():
        print(k, ':', v)
    return


def plot_graph(g, labels=None, graph_layout='shell',
               node_size=200, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):
         
    plt.figure(1)
    plt.title('Graficzne przedstawienie grafu')
    
    G = nx.Graph()
    
    for k, v in g.items():
        G.add_node(k)
        
    for k, v in g.items():
        for i in range(0, len(v[1])):
            G.add_edge(k, v[1][i])

    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(G)
    else:
        graph_pos = nx.shell_layout(G)
        
    # draw
    nx.draw_networkx_nodes(G, graph_pos, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G, graph_pos, width=edge_tickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos, font_size=node_text_size,
                            font_family=text_font)
           
    plt.show()
    return


def plot_graph_degree(g, m_0, m):
    
    max_degree = 0
    for k, v in g.items():
        if len(v[1]) > max_degree:
            max_degree = len(v[1])
    
    plt.figure(2)
    plt.title('Rozkład stopni wierzchołków P(k) w sieci BA z parametrem m=%d' % m)
    plt.ylabel('P(k)')
    plt.xlabel('k')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    
    # rozklad referencyjny:
    x = np.arange(1, max_degree+1)
    y = [0 for i in range(max_degree)]
    for x_i in x:
        y[x_i - 1] = (2 * m * m / (x_i * x_i * x_i))
    plt.plot(x, y, label="rozkład referencyjny", linewidth=2)
    
    # rozklad rzeczywisty:
    # TODO
    
    # pokaz:
    plt.legend(loc=1)
    plt.show()
    return

if __name__ == "__main__":
    main()
