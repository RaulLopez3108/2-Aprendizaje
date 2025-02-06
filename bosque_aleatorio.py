
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

def predice_bosque():
    pass


