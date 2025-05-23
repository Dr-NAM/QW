from __future__ import annotations  # Habilita anotaciones de tipos posteriores (útil para versiones anteriores de Python)
from typing import Callable  # Permite declarar funciones como tipos (ej: funciones que retornan operadores)
from enum import Enum  # Soporte para enumeraciones (útil para manejar modos de direccionamiento)
from dataclasses import dataclass  # Permite definir clases de datos de forma concisa

import numpy as np  # Librería principal para cálculos numéricos y álgebra lineal
import networkx as nx  # Librería para crear y manejar grafos
import copy  # Permite hacer copias profundas de estructuras (como grafos)

# Enumeración para definir cómo se direccionan las aristas al transformar el grafo
# Esto es útil cuando queremos saber si la dirección es desde un nodo ORIGIN o hacia un TARGET
class AddressingType(Enum):
    ORIGIN = 0
    TARGET = 1

# --- DEFINICIÓN DE OPERADORES COIN (Moneda cuántica) ---
# En caminatas cuánticas, el operador moneda determina cómo se modifican las amplitudes antes del desplazamiento

def coins():
    return None  # Esta función no tiene propósito actual, posiblemente placeholder

# Operador Pauli-X: intercambia |0⟩ y |1⟩ (compuerta NOT cuántica)
def X():
    return np.array([[0,1],[1,0]])

# Operador Pauli-Z: mantiene |0⟩ pero cambia el signo de |1⟩ (aplica una fase)
def Z():
    return np.array([[1,0],[0,-1]])

# Identidad 2x2: no cambia el estado sobre el que actúa
def I():
    return np.eye(2)

# --- DEFINICIÓN DE UNA ETAPA EN LA EVOLUCIÓN CUÁNTICA ---
# Cada etapa se define con una etiqueta y una función que construye su operador unitario
@dataclass
class PipeLine:
    label: str  # Nombre descriptivo (por ejemplo: 'oracle', 'coin', 'scatter')
    callback: Callable[[QWSearch], np.ndarray]  # Función que retorna una matriz unitaria al recibir el objeto QWSearch

# --- CLASE PRINCIPAL DE CAMINATA CUÁNTICA ---
class QWSearch:
    def __init__(self, G: nx.Graph, starify: bool = False, addressing: AddressingType = AddressingType.ORIGIN):
        # Si se activa la opción 'starify', se transforma el grafo añadiendo nodos virtuales (uno por cada nodo real)
        if starify:
            G = QWSearch._starify(G)

        self._G = G  # Grafo utilizado para la caminata
        self._edges = list(G.edges())  # Lista de aristas del grafo
        self._index = {e: i for i, e in enumerate(self._edges)}  # Mapa de aristas a su índice único
        self._addressing = addressing  # Tipo de direccionamiento usado

        # Estado cuántico inicial: vector de dimensión 2*|E| (por cada arista hay dos polaridades: + y -)
        self._state = np.zeros(2 * len(self._edges))  # Vector lleno de ceros
        self._state[:] = 1 / np.sqrt(len(self._edges) * 2)  # Estado uniforme: todas las posiciones tienen la misma probabilidad

    def edges(self):
        return self._edges  # Retorna la lista de aristas del grafo

    def nodes(self):
        return self._G.nodes()  # Retorna los nodos del grafo

    def graph(self):
        return self._G  # Retorna el objeto grafo completo

    # Ejecuta la caminata cuántica durante un número de pasos 'ticks'
    def run(self, pipeline: list[PipeLine], ticks: int = 1) -> list[float]:
        U = np.eye(2 * len(self._edges))  # Se parte desde el operador identidad
        for pipe in pipeline:
            U = pipe.callback(self) @ U  # Se compone con cada etapa del pipeline (de derecha a izquierda)

        res = []  # Lista para registrar la probabilidad total en cada paso
        for _ in range(ticks):
            self._state = U @ self._state  # Evoluciona el estado con el operador total
            res.append(np.sum(np.abs(self._state)**2))  # Se guarda la probabilidad total (debe ser ≈ 1 en sistemas cerrados)
        return res  # Devuelve la evolución de la probabilidad total en cada paso

    # Calcula el mejor tiempo de acierto y su probabilidad máxima tras aplicar el pipeline
    def get_T_P(self, pipeline: list[PipeLine], waiting: int = 0) -> tuple[int, float]:
        U = np.eye(2 * len(self._edges))  # Operador identidad como base
        for pipe in pipeline:
            U = pipe.callback(self) @ U  # Se compone el operador final

        max_p = 0  # Probabilidad máxima observada
        max_t = 0  # Paso en el que se alcanza la máxima probabilidad
        state = copy.deepcopy(self._state)  # Estado cuántico inicial (copia independiente)
        for i in range(waiting):  # Se simula hasta 'waiting' pasos
            state = U @ state
            p = np.sum(np.abs(state)**2)
            if p > max_p:
                max_p = p
                max_t = i
        return max_t, max_p  # Retorna (tiempo óptimo, probabilidad máxima)

    # Función auxiliar que transforma el grafo G en su versión starificada
    @staticmethod
    def _starify(G: nx.Graph) -> nx.Graph:
        G2 = copy.deepcopy(G)  # Crea una copia profunda del grafo original
        for u in list(G.nodes):
            virtual = f"v_{u}"  # Se crea un nodo virtual asociado a cada nodo real
            G2.add_node(virtual)  # Se añade el nodo virtual al grafo
            G2.add_edge(u, virtual)  # Se conecta el nodo real con su virtual
        return G2  # Retorna el nuevo grafo starificado

# --- PIPELINE PARA BÚSQUEDA DE NODOS USANDO CAMINATA SOBRE ARISTAS ---
def search_virtual_edges(coin: np.ndarray, scattering: str, targets: list, neg_coin: np.ndarray) -> list[PipeLine]:

    # Oráculo: invierte el signo de las amplitudes sobre aristas que tocan nodos objetivo
    def oracle(qw: QWSearch):
        U = np.eye(2 * len(qw.edges()))  # Matriz identidad del tamaño del espacio de estados
        for e in qw.edges():
            if e[1] in targets:  # Si el nodo destino es un nodo marcado
                i = qw._index[e]  # Se obtiene el índice asociado a la arista
                U[2*i:2*i+2, 2*i:2*i+2] = neg_coin  # Se aplica el operador neg_coin a esa sección del estado
        return U  # Retorna el operador unitario del oráculo

    # Aplica la misma moneda (coin) en todas las aristas
    def apply_coin(qw: QWSearch):
        U = np.eye(2 * len(qw.edges()))
        for e in qw.edges():
            i = qw._index[e]
            U[2*i:2*i+2, 2*i:2*i+2] = coin
        return U  # Devuelve el operador moneda global para todo el grafo

    # Dispersión estilo Grover: opera en los vecindarios locales de cada nodo
    def grover_scatter(qw: QWSearch):
        U = np.eye(2 * len(qw.edges()))
        for n in qw.nodes():
            edges = [e for e in qw.edges() if e[0] == n]  # Se obtienen las aristas incidentes desde n
            if not edges: continue
            m = len(edges)
            Gm = 2 / m * np.ones((m, m)) - np.eye(m)  # Operador de difusión tipo Grover
            block = np.kron(Gm, np.eye(2))  # Se extiende a polaridades con producto de Kronecker
            idxs = [qw._index[e] for e in edges]
            for i in range(len(idxs)):
                for j in range(len(idxs)):
                    U[2*idxs[i]:2*idxs[i]+2, 2*idxs[j]:2*idxs[j]+2] = block[2*i:2*i+2, 2*j:2*j+2]
        return U  # Devuelve el operador de dispersión total

    # Devuelve la secuencia: oráculo → moneda → dispersión
    return [
        PipeLine("oracle", oracle),
        PipeLine("coin", apply_coin),
        PipeLine("scatter", grover_scatter),
    ]
