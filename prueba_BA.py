import utileria as ut
import bosque_aleatorio as ba
import os
import random
__author__ = "Raul Lopez"
__date__ = "enero 2025"

# Descarga y descomprime los datos

url = "https://archive.ics.uci.edu/static/public/373/drug+consumption+quantified.zip"
archivo = "datos/drugs.zip"
archivo_datos = "drug_consumption.data"
atributos = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 13)]

# Descarga datos
if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

# Extrae datos y convierte a numéricos
datos = ut.lee_csv(
    archivo_datos,
    atributos=atributos,
    separador=","
)


# Selecciona los atributos
target = 'Diagnosis'
atributos = [target] + [f'feature_{i}' for i in range(1, 13)]

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Para diferentes cantidades de árboles
errores = []
for n_arboles in [10, 50, 100, 200]:
    bosque = ba.entrena_bosque(
        datos_entrenamiento,
        target,
        atributos,
        n_arboles=n_arboles
    )
    error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)
    errores.append((n_arboles, error_en_muestra, error_en_validacion))

# Muestra los errores
print('n_arboles'.center(10) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 40)
for n_arboles, error_entrenamiento, error_validacion in errores:
    print(
        f'{n_arboles}'.center(10)
        + f'{error_entrenamiento:.2f}'.center(15)
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 40 + '\n')

# Entrena con la mejor cantidad de árboles
bosque = ba.entrena_bosque(datos, target, atributos, n_arboles=50)
error = ba.evalua_bosque(bosque, datos_entrenamiento, target)
print(f'Error del modelo seleccionado entrenado con TODOS los datos: {error:.2f}')
ba.imprime_bosque(bosque)
