import os
import numpy as np
from PIL import Image

################################################################

directorio_actual = os.getcwd()

# Funcion para generar, guardar y renombrar las mascaras necesarias:
def guardar_imagen(archivo_npy):
    # Cargar el archivo npy
    matriz_tridimensional = np.load(archivo_npy)

    # Convertir la matriz tridimensional en una matriz bidimensional sin capas
    matriz_sin_capas = np.max(matriz_tridimensional * np.array([1, 2, 3, 4]), axis=-1)

    # Crear máscaras para cada valor
    mascara_1 = matriz_sin_capas == 1
    mascara_2 = matriz_sin_capas == 2
    mascara_3 = matriz_sin_capas == 3
    mascara_4 = matriz_sin_capas == 4

    # Establecer colores para cada valor
    color_1 = [60, 16, 152]  # Naranja
    color_2 = [132, 41, 246]  # Gris
    color_3 = [110, 193, 228]  # Azul
    color_4 = [254, 221, 58]  # Verde
    color_default = [155, 155, 155]  # Blanco

    # Crear una matriz RGB y asignar colores
    matriz_rgb = np.full((matriz_sin_capas.shape[0], matriz_sin_capas.shape[1], 3), color_default, dtype=np.uint8)
    matriz_rgb[mascara_1] = color_1
    matriz_rgb[mascara_2] = color_2
    matriz_rgb[mascara_3] = color_3
    matriz_rgb[mascara_4] = color_4

    # Guardar la imagen como PNG
    imagen = Image.fromarray(matriz_rgb)
    imagen.save(os.path.join(directorio_actual + "\\dataset\\Tile 1\\masks", archivo_npy[:-4] + ".png"))

################################################################

# Cambiar al directorio de datos
os.chdir(directorio_actual + "\\dataset\\Tile 1\\mask_npy")

# Listar todos los archivos npy en el directorio
archivos_npy = [archivo for archivo in os.listdir() if archivo.endswith(".npy")]

# Iterar sobre cada archivo npy
for archivo_npy in archivos_npy:
    # Guardar la imagen
    guardar_imagen(archivo_npy)

################################################################

directorio_imagenes = directorio_actual + "\\dataset\\Tile 1\\masks"

# Prefijo y sufijo para el nuevo nombre de las imágenes
prefijo = 'image_part_'

# Enumerar archivos en el directorio
for id, nombre_archivo in enumerate(os.listdir(directorio_imagenes)):
    # Verificar si el archivo es una imagen PNG
    if nombre_archivo.lower().endswith('.png'):
        # Crear el nuevo nombre con formato de tres dígitos
        nuevo_nombre = f'{prefijo}{id+1:03d}.png'

        # Construir la ruta completa del archivo antiguo y nuevo
        ruta_antiguo = os.path.join(directorio_imagenes, nombre_archivo)
        ruta_nuevo = os.path.join(directorio_imagenes, nuevo_nombre)

        # Renombrar el archivo
        os.rename(ruta_antiguo, ruta_nuevo)

print("¡Renombrado completado!")

################################################################