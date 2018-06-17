# Core

Base class files.

- `__init__.py`: collects all classes in this folder, so that scripts in other folders can import classes from it (e.g. `from core import BaseModel`). 
- `model.py`: definition for `BaseModel`, which every model should inherit.
- `datasource.py`: definition for `BaseDataSource`, which every data source should inherit.