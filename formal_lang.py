import matplotlib.pyplot as plt
import networkx as nx
import time
from collections import defaultdict
import numpy as np

def is_language_non_empty(grammar, verbose=True):
    """
    Проверяет, существует ли язык заданной грамматики (не пуст ли он).
    
    Алгоритм определяет продуктивные нетерминалы и визуализирует процесс с описаниями.
    
    Параметры:
        grammar (dict): Граматика в виде словаря с ключами:
            'V_T' - множество терминалов (set)
            'V_N' - множество нетерминалов (set)
            'P' - правила продукций (list of tuples: (нетерминал, строка))
            'S' - стартовый нетерминал (str)
        verbose (bool): Выводить ли подробный отчет
    
    Возвращает:
        tuple: (bool, dict) - (непустота языка, словарь с результатами)
    """
    start_time = time.time()
    N_prev = set()
    N_current = set()
    all_non_terminals = grammar['V_N']
    all_terminals = grammar['V_T']
    productions = grammar['P']
    start_symbol = grammar['S']
    iterations = 0
    
    # Подготовка данных для визуализации
    G = nx.DiGraph()
    for nt in all_non_terminals:
        G.add_node(nt)
    edge_weights = defaultdict(int)
    for (A, prod) in productions:
        for symbol in prod:
            if symbol in all_non_terminals:
                edge_weights[(A, symbol)] += 1
                G.add_edge(A, symbol)
    
    # Список для хранения состояний и изменений
    states = [set()]
    changes = [set()]  # Новые нетерминалы на каждой итерации
    
    changed = True
    while changed:
        changed = False
        iterations += 1
        new_non_terminals = set()
        for (A, production) in productions:
            if A not in N_current and all(
                (symbol in all_terminals) or (symbol in N_prev)
                for symbol in production
            ):
                N_current.add(A)
                new_non_terminals.add(A)
                changed = True
        N_prev = N_current.copy()
        states.append(N_current.copy())
        changes.append(new_non_terminals)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Подробный отчет
    productive = N_current
    non_productive = all_non_terminals - productive
    unreachable = find_unreachable_non_terminals(grammar)
    
    results = {
        'productive': productive,
        'non_productive': non_productive,
        'unreachable': unreachable,
        'iterations': iterations,
        'time': execution_time
    }
    
    if verbose:
        print(f"\nРезультаты для грамматики:")
        print(f"Язык существует: {start_symbol in productive}")
        print(f"Продуктивные нетерминалы: {productive}")
        print(f"Непродуктивные нетерминалы: {non_productive}")
        print(f"Недостижимые нетерминалы: {unreachable}")
        print(f"Итераций: {iterations}")
        print(f"Время выполнения: {execution_time:.6f} сек")
    
    # Визуализация графа с описаниями
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    for i, (state, change) in enumerate(zip(states, changes)):
        plt.clf()
        node_colors = ['green' if node in state else 'red' for node in G.nodes]
        edge_widths = [edge_weights.get((u, v), 1) * 2 for u, v in G.edges]
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=600,
                font_size=10, font_weight='bold', edge_color='gray', width=edge_widths)
        
        # Описание окна
        status = "Ожидается завершение" if i < len(states)-1 else ("Язык существует" if start_symbol in state else "Язык пуст")
        description = (
            f"Итерация {i}\n"
            f"Продуктивные нетерминалы: {state}\n"
            f"Добавлены: {change}\n"
            f"Статус: {status}"
        )
        plt.text(0.02, 0.98, description, transform=plt.gca().transAxes, 
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.title(f"Анализ продуктивности нетерминалов: Итерация {i}")
        plt.pause(1.5)
    plt.show()
    
    return start_symbol in productive, results

def find_unreachable_non_terminals(grammar):
    """
    Определяет недостижимые нетерминалы.
    
    Возвращает:
        set: Множество недостижимых нетерминалов
    """
    reachable = {grammar['S']}
    stack = [grammar['S']]
    productions = grammar['P']
    
    while stack:
        current = stack.pop()
        for (A, prod) in productions:
            if A == current:
                for symbol in prod:
                    if symbol in grammar['V_N'] and symbol not in reachable:
                        reachable.add(symbol)
                        stack.append(symbol)
    
    return grammar['V_N'] - reachable

# Пример использования
if __name__ == "__main__":
    # Грамматика арифметических выражений
    grammar_arith = {
        'V_T': {'n', '+', '*', '(', ')'},  # Числа, операции, скобки
        'V_N': {'E', 'T', 'F', 'S', 'U'},  # E: выражение, T: терм, F: фактор, S: старт, U: неиспользуемый
        'P': [
            ('S', 'E'),          # Старт порождает выражение
            ('E', 'E+T'),        # Сложение
            ('E', 'T'),          # Терм
            ('T', 'T*F'),        # Умножение
            ('T', 'F'),          # Фактор
            ('F', '(E)'),        # Скобки
            ('F', 'n'),          # Число
            ('U', 'U'),          # Непродуктивный нетерминал
        ],
        'S': 'S'
    }
    
    is_language_non_empty(grammar_arith)