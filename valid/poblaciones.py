from pathlib import Path
import csv

DATA_DIR = Path("data")
CSV_PREFIX = "poblacion-"
TOTAL_BARRAS = 40

def mostrar_grafico_ascii(categorias):
    max_porcentaje = max(p for _, p in categorias)
    base_max_barras = TOTAL_BARRAS
    max_barras = base_max_barras + 10 

    for categoria, porcentaje in categorias:
        barras = round((porcentaje / max_porcentaje) * max_barras)
        print(f"{categoria:25}: {'|' * barras:<{max_barras}} {porcentaje:6.2f}%")


def validar_archivos_poblacion():
    archivos = sorted(DATA_DIR.glob(f"{CSV_PREFIX}*.csv"))

    if not archivos:
        print("No se encontraron archivos de población.")
        return

    for archivo in archivos:
        print(f"\nArchivo: {archivo.name}")
        total = 0.0
        categorias = []
        
        with archivo.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                categoria = row['categoria'].strip()
                porcentaje = float(row['porcentaje'])
                total += porcentaje
                categorias.append((categoria, porcentaje))
        
        categorias.sort()
        mostrar_grafico_ascii(categorias)

        if abs(total - 100.0) > 0.01:
            print(f"\nERROR: La suma total es {total:.2f}%, debe ser 100.00%\n")
        else:
            print(f"\nTotal correcto: {total:.2f}%\n")

if __name__ == "__main__":
    validar_archivos_poblacion()