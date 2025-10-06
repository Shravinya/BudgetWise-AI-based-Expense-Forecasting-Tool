# src/utils/file_operations.py
import os
import pickle
import json
import pandas as pd
from src.utils.exception import BudgetException
import sys

def save_object(file_path, obj):
    """
    Save any Python object (model, transformer, encoder, etc.)
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise BudgetException(f"Error saving object: {file_path} - {e}", sys)


def load_object(file_path):
    """
    Load a Python object from file.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise BudgetException(f"Error loading object: {file_path} - {e}", sys)


def save_json(file_path, data):
    """
    Save a dictionary as JSON.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise BudgetException(f"Error saving JSON: {file_path} - {e}", sys)


def load_json(file_path):
    """
    Load JSON file into dictionary.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise BudgetException(f"Error loading JSON: {file_path} - {e}", sys)


def save_dataframe(file_path, df: pd.DataFrame):
    """
    Save DataFrame as CSV.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise BudgetException(f"Error saving DataFrame: {file_path} - {e}", sys)
