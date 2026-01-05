# charts/base_chart.py
from abc import ABC, abstractmethod
import streamlit as st

class BaseChart(ABC):
    """Clase base para todos los tipos de gráficas"""
    
    @abstractmethod
    def render_parameters(self) -> dict:
        """Renderiza los parámetros específicos de la gráfica y devuelve sus valores"""
        pass
    
    @abstractmethod 
    def create_chart(self, df, main_variable: str, extra_variable: str, **params):
        """Crea la gráfica con los parámetros dados"""
        pass