# utils/data_validator.py
import pandas as pd
from typing import Dict, List, Any

# Columnas obligatorias
REQUIRED_COLUMNS = [
    'Company name Latin alphabet',
    'Country ISO code', 
    'Website address'
]

# Rangos esperados
MIN_ROWS = 1000
MAX_ROWS = 10000
MIN_COLUMNS = 140
MAX_COLUMNS = 170

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida que el DataFrame tenga la estructura esperada para datos financieros
    """
    errors = []
    warnings = []
    
    # Validación básica: DataFrame vacío
    if df.empty:
        errors.append("DataFrame is empty")
        return {'is_valid': False, 'errors': errors, 'warnings': warnings}
    
    # Validar dimensiones
    rows, cols = df.shape
    
    if rows < MIN_ROWS:
        warnings.append(f"Dataset has {rows:,} rows, expected at least {MIN_ROWS:,}")
    elif rows > MAX_ROWS:
        warnings.append(f"Dataset has {rows:,} rows, expected at most {MAX_ROWS:,}")
    
    if cols < MIN_COLUMNS:
        warnings.append(f"Dataset has {cols} columns, expected around {MIN_COLUMNS}-{MAX_COLUMNS}")
    elif cols > MAX_COLUMNS:
        warnings.append(f"Dataset has {cols} columns, expected around {MIN_COLUMNS}-{MAX_COLUMNS}")
    
    # Validar columnas obligatorias
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Validaciones específicas de las columnas obligatorias
    if 'Company name Latin alphabet' in df.columns:
        null_companies = df['Company name Latin alphabet'].isnull().sum()
        if null_companies > 0:
            warnings.append(f"{null_companies} companies have missing names")
    
    if 'Country ISO code' in df.columns:
        null_countries = df['Country ISO code'].isnull().sum()
        if null_countries > 0:
            warnings.append(f"{null_countries} companies have missing country codes")
        
        # Validar formato de códigos ISO (2-3 caracteres)
        if not df['Country ISO code'].isnull().all():
            invalid_codes = df['Country ISO code'].dropna().str.len().apply(lambda x: x < 2 or x > 3).sum()
            if invalid_codes > 0:
                warnings.append(f"{invalid_codes} invalid country ISO codes found")
    
    if 'Website address' in df.columns:
        null_websites = df['Website address'].isnull().sum()
        if null_websites > 0:
            warnings.append(f"{null_websites} companies have missing website addresses")
    
    # Validar duplicados en nombres de compañías
    if 'Company name Latin alphabet' in df.columns:
        duplicated_names = df['Company name Latin alphabet'].duplicated().sum()
        if duplicated_names > 0:
            warnings.append(f"{duplicated_names} duplicate company names found")
    
    return {
        'is_valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }