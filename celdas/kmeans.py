import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator  # Para detectar el codo automáticamente

# ============================================
# FUNCIÓN DE LIMPIEZA DE COORDENADAS (MEJORADA)
# ============================================
def limpiar_coordenadas_miles(df, lat_col='lat', lon_col='lon'):
    """
    Limpia coordenadas que pueden tener problemas de formato.
    Maneja separadores decimales y formatos incorrectos.
    """
    def limpiar_numero(numero):
        if pd.isna(numero):
            return np.nan
        
        numero_str = str(numero).strip()
        
        # Remover separadores de miles
        numero_str = numero_str.replace(',', '').replace(' ', '')
        
        # Caso especial para longitud negativa (ej: -76.532154)
        if numero_str.startswith("-7") and len(numero_str) > 3:
            # Si no tiene punto, insertar después del tercer dígito
            if '.' not in numero_str:
                numero_str = numero_str[0:3] + '.' + numero_str[3:]
        # Caso especial para latitud positiva (ej: 3.451234)
        elif numero_str.startswith("3") and len(numero_str) > 1:
            if '.' not in numero_str:
                numero_str = numero_str[0:1] + '.' + numero_str[1:]
        
        try:
            return float(numero_str)
        except ValueError:
            return np.nan
    
    # Aplicar limpieza
    df[lat_col] = df[lat_col].apply(limpiar_numero)
    df[lon_col] = df[lon_col].apply(limpiar_numero)
    
    # Eliminar NaN
    antes = len(df)
    df = df.dropna(subset=[lat_col, lon_col])
    despues = len(df)
    
    if antes > despues:
        print(f"⚠️  Se eliminaron {antes - despues} filas con coordenadas inválidas")
    
    return df

# ============================================
# FUNCIÓN PARA DETECTAR COLUMNAS DE COORDENADAS
# ============================================
def find_coord_cols(df):
    """Detecta automáticamente las columnas de latitud y longitud."""
    candidates = [
        ("long", "lat"), ("lon", "lat"), ("x", "y"), ("X", "Y"),
        ("longitude", "latitude"), ("LONG", "LAT")
    ]
    
    cols = set(df.columns)
    
    # Búsqueda exacta
    for lonc, latc in candidates:
        if lonc in cols and latc in cols:
            return lonc, latc
    
    # Búsqueda por substring (case-insensitive)
    lon_col = None
    lat_col = None
    
    for c in df.columns:
        name = str(c).lower()
        if ("lon" in name or "long" in name or "lng" in name or name == "x") and lon_col is None:
            lon_col = c
        if ("lat" in name or name == "y") and lat_col is None:
            lat_col = c
    
    if lon_col is not None and lat_col is not None:
        return lon_col, lat_col
    
    print("❌ No se pudieron detectar las columnas de coordenadas.")
    print(f"Columnas disponibles: {list(df.columns)}")
    return None, None

# ============================================
# 1️⃣ CARGAR Y PROCESAR DATOS
# ============================================
print("=" * 70)
print("📊 OPTIMIZACIÓN DE TAMAÑO DE CELDA CON K-MEANS")
print("=" * 70)

# Cargar datos
try:
    df = pd.read_csv("violencia-db/SEMANA.csv")
    print(f"✅ Archivo cargado: {len(df)} registros")
except FileNotFoundError:
    print("⚠️  Archivo no encontrado. Generando datos simulados...")
    # Datos simulados para Cali (lat: 3.4, lon: -76.5)
    np.random.seed(42)
    n_points = 2000
    df = pd.DataFrame({
        'lat': np.random.normal(3.45, 0.1, n_points),
        'lon': np.random.normal(-76.53, 0.1, n_points)
    })
    print(f"✅ Datos simulados generados: {len(df)} puntos")

# Detectar columnas de coordenadas
lon_col, lat_col = find_coord_cols(df)

if lon_col is None or lat_col is None:
    print("❌ Error: No se encontraron columnas de coordenadas válidas")
    exit()

print(f"📍 Columnas detectadas: lon='{lon_col}', lat='{lat_col}'")

# Limpiar coordenadas
df = limpiar_coordenadas_miles(df, lat_col, lon_col)

# Convertir a numpy array
coords = df[[lon_col, lat_col]].astype(float).to_numpy()
print(f"✅ Puntos válidos para análisis: {coords.shape[0]}\n")

# Validar rango de coordenadas (para Cali debería estar cerca de lat:3.4, lon:-76.5)
print(f"📌 Rango de coordenadas:")
print(f"   Latitud:  {coords[:, 1].min():.6f} a {coords[:, 1].max():.6f}")
print(f"   Longitud: {coords[:, 0].min():.6f} a {coords[:, 0].max():.6f}\n")

# ============================================
# 2️⃣ MÉTODO DEL CODO - BÚSQUEDA DE K ÓPTIMO
# ============================================
print("🔍 Calculando método del codo...")
inercias = []
silhouettes = []
ks = range(10, 201, 10)  # De 10 a 200 clusters, paso de 10

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(coords)
    inercias.append(kmeans.inertia_)
    print(f"   K={k:3d} → Inercia={kmeans.inertia_:.4f}")

print("\n✅ Cálculo completado\n")

# ============================================
# 3️⃣ DETECTAR EL CODO AUTOMÁTICAMENTE
# ============================================
try:
    # Usar KneeLocator para encontrar el codo
    kn = KneeLocator(list(ks), inercias, curve='convex', direction='decreasing')
    K_opt = kn.knee if kn.knee is not None else 90
    print(f"🎯 Codo detectado automáticamente en K = {K_opt}")
except:
    # Fallback: calcular segunda derivada manualmente
    second_diff = np.diff(inercias, 2)
    K_opt = ks[np.argmax(second_diff) + 1]
    print(f"🎯 Codo estimado (método alternativo) en K = {K_opt}")

# ============================================
# 4️⃣ CALCULAR TAMAÑO DE CELDA ÓPTIMO
# ============================================
area_cali = 619  # km²
area_celda = area_cali / K_opt
lado_celda_km = np.sqrt(area_celda)
lado_celda_m = lado_celda_km * 1000

print("\n" + "=" * 70)
print("🏆 RESULTADOS - MÉTODO K-MEANS")
print("=" * 70)
print(f"✅ Número óptimo de clusters (K): {K_opt}")
print(f"📐 Área promedio de celda: {area_celda:.2f} km²")
print(f"📏 Lado promedio de celda: {lado_celda_km:.2f} km ({lado_celda_m:.0f} m)")
print(f"📊 Puntos promedio por cluster: {len(coords) / K_opt:.1f}")
print("=" * 70)