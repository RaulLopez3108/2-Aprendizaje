import random
import arboles_numericos as an
from collections import Counter
__author__ = "Raul Lopez"
__date__ = "enero 2025"


def entrena_bosque(datos,target,clase_default,
                   max_profundidad=None,n_arboles=100,  
                   variables_seleccionadas = None):
    bosque = []
    random.seed(42)
    subconjunto = random.choices(datos, k = len(datos))


    for _ in range(n_arboles):               
        arbol = an.entrena_arbol(
            datos = subconjunto,
            target = target,
            clase_default = clase_default,
            max_profundidad=max_profundidad,            
            variables_seleccionadas=variables_seleccionadas
        )
        bosque.append(arbol)
    return bosque


def predice_bosque(bosque, datos):
    predicciones = []
    for arbol in bosque:
        predicciones.append(an.predice_arbol(arbol, datos))
    return Counter(predicciones).most_common(1)[0][0]

def evaula_bosque(bosque,datos,target):
    predicciones = predice_bosque(bosque, datos)
    return sum(1 for p, d in zip(predicciones, datos) if p == d[target]) / len(datos)

