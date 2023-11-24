import os

################################################################

directorio_actual = os.getcwd()

directorio_imagenes = directorio_actual + "\\dataset\\Tile 1\\images"

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
