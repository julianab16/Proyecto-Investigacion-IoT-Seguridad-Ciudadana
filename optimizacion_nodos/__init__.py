"""
Módulo de Optimización de Nodos LoRaWAN con NSGA-II
====================================================

Este módulo contiene herramientas para optimizar la configuración
de nodos IoT LoRaWAN usando el algoritmo multiobjetivo NSGA-II.

Objetivos:
  - MAX PDR(SF, TP, λ)      [Packet Delivery Rate]
  - MIN Energy(SF, TP, λ)   [Consumo Energético]
  - MIN Latency(SF, λ)      [Latencia]

Módulos principales:
  - nsga.py              : Implementación del algoritmo NSGA-II
  - ejemplos_uso.py      : Ejemplos prácticos de uso
  - verificar_instalacion.py : Script de validación

Documentación:
  - NSGA2_DOCUMENTATION.md   : Documentación técnica
  - ECUACIONES_RESUMIDAS.md  : Ecuaciones del modelo
  - README.md                : Guía rápida

Uso:
  python nsga.py              # Ejecutar optimización
  python ejemplos_uso.py      # Ver ejemplos
  python verificar_instalacion.py  # Validar instalación
"""

__version__ = "1.0"
__author__ = "IoT Seguridad Ciudadana"
__date__ = "March 20, 2026"

__all__ = ['nsga', 'ejemplos_uso', 'verificar_instalacion']
