
import random
import arboles_numericos

def entrena_bosque(datos,target,clase_default,
                   max_profundidad=None,n_arboles=100,
                   acc_nodo=1.0,min_ejemplos=0,frac_datos=0.8,
                   frac_variables=0.8):
    bosque = []

    for _ in range(n_arboles):
        datos_arbol = datos.sample(frac= frac_datos,replace = True)

        variables_seleccionadas = random.sample(
            list(datos_arbol.columns),
            int(frac_variables * len(datos_arbol.columns))
        )

        arbol = arboles_numericos.entrena_arbol(
            datos_arbol[variables_seleccionadas],
            target,
            clase_default,
            max_profundidad=max_profundidad,
            acc_nodo=acc_nodo,
            min_ejemplos=min_ejemplos,
            variables_seleccionadas=variables_seleccionadas
        )
        bosque.append(arbol)
    return bosque

def predice_bosque(bosque,datos):
    return[bosque.predice(d)for d in datos]


def predice():
    if arboles_numericos.self.terminal:
            return arboles_numericos.self.clase_default               
    if arboles_numericos.instancia[arboles_numericos.self.atributo] < arboles_numericos.self.valor:
        return arboles_numericos.self.hijo_menor.predice(arboles_numericos.instancia)       
    return arboles_numericos.self.hijo_mayor.predice(arboles_numericos.instancia)


