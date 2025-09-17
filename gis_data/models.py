from django.db import models

# models.py
from django.contrib.gis.db import models
from django.db import models as django_models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone

class Barrio(models.Model):
    """Modelo para los barrios de la ciudad"""
    nombre = django_models.CharField(max_length=100)
    codigo = django_models.CharField(max_length=20, unique=True, null=True, blank=True)
    comuna = django_models.IntegerField(null=True, blank=True, validators=[MinValueValidator(1)])
    cuadrante = django_models.CharField(max_length=10, null=True, blank=True)
    geometria = models.MultiPolygonField(srid=4326)
    area_m2 = django_models.FloatField(null=True, blank=True, help_text="Área en metros cuadrados")
    created_at = django_models.DateTimeField(default=timezone.now)
    
    class Meta:
        verbose_name = "Barrio"
        verbose_name_plural = "Barrios"
        ordering = ['nombre']
    
    def __str__(self):
        return self.nombre


class Homicidio(models.Model):
    """Modelo principal para registros de homicidios"""
    GENERO_CHOICES = [
        ('M', 'Masculino'),
        ('F', 'Femenino'),
        ('O', 'Otro'),
    ]

    VIOLENCIA_TIPO_CHOICES = [
        ('CONVIVENCIA', 'Convivencia'),
        ('FEMINICIDIO', 'Feminicidio'),
        ('DELINCUENCIA', 'Delincuencia'),
        ('VIOLENCIA DE GENERO', 'Violencia de Género'),
        ('OTRO', 'Otro'),
    ]
    
    codigo_caso = django_models.CharField(max_length=50, unique=True, null=True, blank=True)
    fecha = django_models.DateField()
    hora = django_models.TimeField(null=True, blank=True)
    dia_de_semana = django_models.CharField(max_length=20, blank=True)
    tipo_violencia = django_models.CharField(max_length=100, blank=True)
    ubicacion = models.PointField(srid=4326)
    lugar_hechos = django_models.CharField(max_length=200, blank=True)
    barrio = django_models.ForeignKey(
        Barrio, 
        on_delete=django_models.CASCADE,
        null=True, 
        blank=True
    )
    direccion = django_models.TextField(blank=True)
    modalidad = django_models.CharField(max_length=100, blank=True)
    arma_utilizada = django_models.CharField(max_length=50, blank=True)
    genero_victima = django_models.CharField(
        max_length=1, 
        choices=GENERO_CHOICES,
        null=True, 
        blank=True
    )
    edad_victima = django_models.IntegerField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(0), MaxValueValidator(120)]
    )
    resumen = django_models.TextField(blank=True)
    created_at = django_models.DateTimeField(default=timezone.now)
    updated_at = django_models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Homicidio"
        verbose_name_plural = "Homicidios"
        ordering = ['-fecha', '-hora']
    
    def __str__(self):
        return f"Homicidio {self.codigo_caso or self.id} - {self.fecha}"


class EstacionPolicia(models.Model):
    """Modelo para estaciones de policía"""
    TIPO_CHOICES = [
        ('CAI', 'Centro de Atención Inmediata'),
        ('ESTACION', 'Estación de Policía'),
        ('SUBESTACION', 'Subestación de Policía'),
        ('PUESTO', 'Puesto de Policía'),
    ]
    
    nombre = django_models.CharField(max_length=100)
    tipo = django_models.CharField(max_length=50, choices=TIPO_CHOICES, blank=True)
    ubicacion = models.PointField(srid=4326)
    direccion = django_models.TextField(blank=True)
    telefono = django_models.CharField(max_length=20, blank=True)
    cobertura_geometria = models.PolygonField(srid=4326, null=True, blank=True)
    created_at = django_models.DateTimeField(default=timezone.now)
    
    class Meta:
        verbose_name = "Estación de Policía"
        verbose_name_plural = "Estaciones de Policía"
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.nombre} ({self.get_tipo_display()})"


class AlumbradoPublico(models.Model):
    """Modelo para puntos de alumbrado público"""
    TIPO_CHOICES = [
        ('LED', 'LED'),
        ('SOLAR', 'Solar'),
    ]
    
    ESTADO_CHOICES = [
        ('FUNCIONANDO', 'Funcionando'),
        ('DAÑADO', 'Dañado'),
        ('APAGADO', 'Apagado'),
        ('MANTENIMIENTO', 'En Mantenimiento'),
    ]
    
    tipo = django_models.CharField(max_length=50, choices=TIPO_CHOICES, blank=True)
    estado = django_models.CharField(max_length=20, choices=ESTADO_CHOICES, default='FUNCIONANDO')
    ubicacion = models.PointField(srid=4326)
    created_at = django_models.DateTimeField(default=timezone.now)
    
    class Meta:
        verbose_name = "Alumbrado Público"
        verbose_name_plural = "Alumbrado Público"
        ordering = ['tipo', 'estado']
    
    def __str__(self):
        return f"Luminaria {self.tipo} - {self.estado}"

"""
class TransportePublico(models.Model):
    TIPO_CHOICES = [
        ('BUS', 'Bus'),
        ('MIO', 'Mio'),
        ('CABLE', 'Cable'),
    ]
    
    tipo = django_models.CharField(max_length=50, choices=TIPO_CHOICES)
    nombre_ruta = django_models.CharField(max_length=100)
    geometria = models.LineStringField(srid=4326)  # LINESTRING para la ruta
    frecuencia_minutos = django_models.IntegerField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(1)]
    )
    created_at = django_models.DateTimeField(default=timezone.now)
    
    class Meta:
        verbose_name = "Transporte Público"
        verbose_name_plural = "Transporte Público"
        ordering = ['tipo', 'nombre_ruta']
    
    def __str__(self):
        return f"{self.get_tipo_display()} - {self.nombre_ruta}"


class ParadaTransporte(models.Model):
    TIPO_CHOICES = [
        ('PARADA', 'Parada'),
        ('ESTACION', 'Estación'),
        ('TERMINAL', 'Terminal'),
    ]
    
    nombre = django_models.CharField(max_length=100)
    tipo = django_models.CharField(max_length=50, choices=TIPO_CHOICES)
    ubicacion = models.PointField(srid=4326)
    rutas_que_pasan = django_models.JSONField(default=list, blank=True)  # Array de rutas
    tiene_iluminacion = django_models.BooleanField(default=False)
    tiene_seguridad = django_models.BooleanField(default=False)
    created_at = django_models.DateTimeField(default=timezone.now)
    
    class Meta:
        verbose_name = "Parada de Transporte"
        verbose_name_plural = "Paradas de Transporte"
        ordering = ['nombre']
    
    def __str__(self):
        return f"{self.nombre} ({self.get_tipo_display()})"
    
    @property
    def rutas_display(self):
        return ", ".join(self.rutas_que_pasan) if self.rutas_que_pasan else "Sin rutas"


# Modelo adicional para relacionar transporte con paradas
class TransporteParada(django_models.Model):
    transporte = django_models.ForeignKey(TransportePublico, on_delete=django_models.CASCADE)
    parada = django_models.ForeignKey(ParadaTransporte, on_delete=django_models.CASCADE)
    tiempo_estimado = django_models.IntegerField(
        null=True, 
        blank=True,
        help_text="Tiempo estimado desde el inicio de la ruta en minutos"
    )
    
    class Meta:
        unique_together = ['transporte', 'parada']
        ordering = ['transporte', 'orden']
        verbose_name = "Relación Transporte-Parada"
        verbose_name_plural = "Relaciones Transporte-Parada"
    
    def __str__(self):
        return f"{self.transporte.nombre_ruta} - {self.parada.nombre}"
"""